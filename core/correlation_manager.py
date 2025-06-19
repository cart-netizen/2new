# core/correlation_manager.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
from scipy import stats
from sklearn.covariance import LedoitWolf
import networkx as nx
import warnings

from core.data_fetcher import DataFetcher
from core.enums import Timeframe
from utils.logging_config import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


class CorrelationType(Enum):
  """Типы корреляций"""
  PEARSON = "pearson"
  SPEARMAN = "spearman"
  KENDALL = "kendall"
  DYNAMIC = "dynamic"


@dataclass
class CorrelationData:
  """Данные о корреляции между активами"""
  symbol1: str
  symbol2: str
  correlation: float
  correlation_type: CorrelationType
  timeframe: Timeframe
  period_days: int
  timestamp: datetime
  p_value: Optional[float] = None
  confidence_interval: Optional[Tuple[float, float]] = None
  is_significant: bool = True
  rolling_std: Optional[float] = None


@dataclass
class PortfolioRiskMetrics:
  """Метрики риска портфеля с учетом корреляций"""
  portfolio_variance: float
  portfolio_volatility: float
  diversification_ratio: float
  concentration_risk: float
  max_correlation: float
  avg_correlation: float
  effective_assets: float  # Эффективное количество независимых активов
  risk_contribution: Dict[str, float]  # Вклад каждого актива в риск


class CorrelationManager:
  """
  Менеджер для анализа корреляций между активами и оптимизации портфеля
  """

  def __init__(self, data_fetcher: DataFetcher):
    self.data_fetcher = data_fetcher

    # Кэш корреляций
    self.correlation_cache: Dict[str, CorrelationData] = {}
    self.cache_ttl = 3600  # 1 час

    # Матрица корреляций
    self.correlation_matrix: Optional[pd.DataFrame] = None
    self.covariance_matrix: Optional[pd.DataFrame] = None

    # Параметры анализа
    self.min_correlation_period = 30  # Минимум 30 дней данных
    self.correlation_threshold = 0.7  # Порог высокой корреляции
    self.p_value_threshold = 0.05  # Уровень значимости

    # График корреляций для визуализации
    self.correlation_graph = nx.Graph()

    # История изменения корреляций
    self.correlation_history: Dict[str, List[CorrelationData]] = {}

  async def analyze_portfolio_correlation(self, symbols: List[str],
                                          timeframe: Timeframe = Timeframe.ONE_HOUR,
                                          lookback_days: int = 30) -> Dict[str, Any]:
    """
    Анализирует корреляции между всеми активами в портфеле
    """
    logger.info(f"Анализ корреляций для {len(symbols)} символов за {lookback_days} дней")

    # 1. Получаем данные для всех символов
    all_returns = await self._get_returns_data(symbols, timeframe, lookback_days)

    if all_returns.empty or len(all_returns.columns) < 2:
      logger.warning("Недостаточно данных для анализа корреляций")
      return {}

    # 2. Рассчитываем различные типы корреляций
    correlations = {
      'pearson': self._calculate_correlation_matrix(all_returns, CorrelationType.PEARSON),
      'spearman': self._calculate_correlation_matrix(all_returns, CorrelationType.SPEARMAN),
      'dynamic': self._calculate_dynamic_correlations(all_returns)
    }

    # 3. Обновляем основную матрицу корреляций
    self.correlation_matrix = correlations['pearson']

    # 4. Рассчитываем ковариационную матрицу с регуляризацией
    self.covariance_matrix = self._calculate_robust_covariance(all_returns)

    # 5. Анализируем кластеры коррелированных активов
    clusters = self._find_correlation_clusters(self.correlation_matrix)

    # 6. Рассчитываем метрики риска портфеля
    risk_metrics = self._calculate_portfolio_risk_metrics(all_returns)

    # 7. Находим проблемные пары с высокой корреляцией
    high_correlations = self._find_high_correlations(self.correlation_matrix)

    # 8. Рекомендации по диверсификации
    recommendations = self._generate_diversification_recommendations(
      correlations, clusters, risk_metrics
    )

    # 9. Обновляем граф корреляций
    self._update_correlation_graph(self.correlation_matrix)

    # 10. Сохраняем в историю
    self._save_correlation_history(symbols, correlations)

    return {
      'correlation_matrices': correlations,
      'clusters': clusters,
      'risk_metrics': risk_metrics,
      'high_correlations': high_correlations,
      'recommendations': recommendations,
      'timestamp': datetime.now()
    }

  async def _get_returns_data(self, symbols: List[str],
                              timeframe: Timeframe,
                              lookback_days: int) -> pd.DataFrame:
    """Получает данные о доходностях для всех символов"""
    all_returns = pd.DataFrame()

    # Параллельная загрузка данных
    tasks = []
    for symbol in symbols:
      task = self._get_symbol_returns(symbol, timeframe, lookback_days)
      tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Объединяем результаты
    for symbol, result in zip(symbols, results):
      if isinstance(result, pd.Series) and not result.empty:
        all_returns[symbol] = result
      else:
        logger.warning(f"Не удалось получить данные для {symbol}")

    # Выравниваем индексы и заполняем пропуски
    all_returns = all_returns.fillna(method='ffill').dropna()

    return all_returns

  async def _get_symbol_returns(self, symbol: str,
                                timeframe: Timeframe,
                                lookback_days: int) -> pd.Series:
    """Получает доходности для одного символа"""
    try:
      # Рассчитываем необходимое количество баров
      bars_per_day = {
        Timeframe.FIVE_MINUTES: 288,
        Timeframe.FIFTEEN_MINUTES: 96,
        Timeframe.THIRTY_MINUTES: 48,
        Timeframe.ONE_HOUR: 24,
        Timeframe.FOUR_HOURS: 6,
        Timeframe.ONE_DAY: 1
      }

      limit = bars_per_day.get(timeframe, 24) * lookback_days

      # Получаем данные
      data = await self.data_fetcher.get_historical_candles(
        symbol, timeframe, limit=min(limit, 1000)
      )

      if data.empty:
        return pd.Series()

      # Рассчитываем логарифмические доходности
      returns = np.log(data['close'] / data['close'].shift(1))
      returns = returns.dropna()

      return returns

    except Exception as e:
      logger.error(f"Ошибка получения данных для {symbol}: {e}")
      return pd.Series()

  def _calculate_correlation_matrix(self, returns: pd.DataFrame,
                                    corr_type: CorrelationType) -> pd.DataFrame:
    """Рассчитывает матрицу корреляций"""
    if corr_type == CorrelationType.PEARSON:
      return returns.corr(method='pearson')
    elif corr_type == CorrelationType.SPEARMAN:
      return returns.corr(method='spearman')
    elif corr_type == CorrelationType.KENDALL:
      return returns.corr(method='kendall')
    else:
      return returns.corr()

  def _calculate_dynamic_correlations(self, returns: pd.DataFrame,
                                      window: int = 30) -> pd.DataFrame:
    """Рассчитывает динамические корреляции (DCC-GARCH упрощенный)"""
    # Скользящие корреляции
    rolling_corr = returns.rolling(window).corr()

    # Берем последнее значение как текущую динамическую корреляцию
    latest_date = returns.index[-1]
    dynamic_corr = pd.DataFrame(index=returns.columns, columns=returns.columns)

    for col1 in returns.columns:
      for col2 in returns.columns:
        if col1 == col2:
          dynamic_corr.loc[col1, col2] = 1.0
        else:
          try:
            dynamic_corr.loc[col1, col2] = rolling_corr.loc[latest_date, col1][col2]
          except:
            dynamic_corr.loc[col1, col2] = returns[col1].corr(returns[col2])

    return dynamic_corr.astype(float)

  def _calculate_robust_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
    """Рассчитывает робастную ковариационную матрицу"""
    try:
      # Используем Ledoit-Wolf shrinkage для регуляризации
      lw = LedoitWolf()
      cov_matrix, _ = lw.fit(returns).covariance_, lw.shrinkage_

      return pd.DataFrame(
        cov_matrix,
        index=returns.columns,
        columns=returns.columns
      )
    except Exception as e:
      logger.error(f"Ошибка расчета робастной ковариации: {e}")
      # Fallback на обычную ковариацию
      return returns.cov()

  def _find_correlation_clusters(self, corr_matrix: pd.DataFrame,
                                 threshold: float = 0.7) -> List[List[str]]:
    """Находит кластеры сильно коррелированных активов"""
    # Создаем граф
    G = nx.Graph()

    # Добавляем ребра для высоких корреляций
    for i in range(len(corr_matrix)):
      for j in range(i + 1, len(corr_matrix)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) >= threshold:
          G.add_edge(
            corr_matrix.index[i],
            corr_matrix.columns[j],
            weight=abs(corr)
          )

    # Находим компоненты связности
    clusters = list(nx.connected_components(G))

    return [list(cluster) for cluster in clusters if len(cluster) > 1]

  def _calculate_portfolio_risk_metrics(self, returns: pd.DataFrame) -> PortfolioRiskMetrics:
    """Рассчитывает метрики риска портфеля"""
    n_assets = len(returns.columns)

    # Равновзвешенный портфель для примера
    weights = np.ones(n_assets) / n_assets

    # Ковариационная матрица
    cov_matrix = returns.cov()

    # Волатильности активов
    asset_vols = returns.std()

    # Портфельная дисперсия и волатильность
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)

    # Средневзвешенная волатильность активов
    weighted_avg_vol = np.dot(weights, asset_vols)

    # Коэффициент диверсификации
    diversification_ratio = weighted_avg_vol / portfolio_volatility

    # Концентрационный риск (Herfindahl index)
    concentration_risk = np.sum(weights ** 2)

    # Корреляционные метрики
    corr_matrix = returns.corr()
    upper_triangle = np.triu(corr_matrix, k=1)
    max_correlation = np.max(np.abs(upper_triangle))
    avg_correlation = np.mean(np.abs(upper_triangle[upper_triangle != 0]))

    # Эффективное количество активов
    eigenvalues = np.linalg.eigvals(corr_matrix)
    effective_assets = np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)

    # Вклад каждого актива в риск
    marginal_contributions = np.dot(cov_matrix, weights) / portfolio_volatility
    risk_contributions = weights * marginal_contributions
    risk_contribution_dict = dict(zip(returns.columns, risk_contributions))

    return PortfolioRiskMetrics(
      portfolio_variance=portfolio_variance,
      portfolio_volatility=portfolio_volatility,
      diversification_ratio=diversification_ratio,
      concentration_risk=concentration_risk,
      max_correlation=max_correlation,
      avg_correlation=avg_correlation,
      effective_assets=effective_assets,
      risk_contribution=risk_contribution_dict
    )

  def _find_high_correlations(self, corr_matrix: pd.DataFrame,
                              threshold: float = None) -> List[Dict[str, Any]]:
    """Находит пары активов с высокой корреляцией"""
    if threshold is None:
      threshold = self.correlation_threshold

    high_correlations = []

    for i in range(len(corr_matrix)):
      for j in range(i + 1, len(corr_matrix)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) >= threshold:
          high_correlations.append({
            'symbol1': corr_matrix.index[i],
            'symbol2': corr_matrix.columns[j],
            'correlation': corr,
            'risk_level': 'high' if abs(corr) > 0.85 else 'medium'
          })

    # Сортируем по убыванию корреляции
    high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

    return high_correlations

  def _generate_diversification_recommendations(self, correlations: Dict,
                                                clusters: List[List[str]],
                                                risk_metrics: PortfolioRiskMetrics) -> Dict[str, Any]:
    """Генерирует рекомендации по диверсификации портфеля"""
    recommendations = {
      'actions': [],
      'warnings': [],
      'optimization_suggestions': []
    }

    # 1. Проверка на чрезмерную концентрацию
    if risk_metrics.concentration_risk > 0.2:
      recommendations['warnings'].append(
        "Высокий риск концентрации. Рассмотрите более равномерное распределение позиций."
      )

    # 2. Проверка на низкую диверсификацию
    if risk_metrics.diversification_ratio < 1.5:
      recommendations['warnings'].append(
        f"Низкий коэффициент диверсификации ({risk_metrics.diversification_ratio:.2f}). "
        f"Портфель ведет себя как {risk_metrics.effective_assets:.1f} независимых активов."
      )

    # 3. Рекомендации по кластерам
    for cluster in clusters:
      if len(cluster) > 2:
        recommendations['actions'].append(
          f"Снизить экспозицию в кластере коррелированных активов: {', '.join(cluster[:3])}..."
        )

    # 4. Проверка максимальной корреляции
    if risk_metrics.max_correlation > 0.9:
      recommendations['warnings'].append(
        f"Обнаружена очень высокая корреляция ({risk_metrics.max_correlation:.2f}). "
        "Рассмотрите замену одного из активов."
      )

    # 5. Оптимизационные предложения
    if risk_metrics.avg_correlation > 0.5:
      recommendations['optimization_suggestions'].append(
        "Добавьте активы с отрицательной или низкой корреляцией (например, стейблкоины или инверсные токены)"
      )

    # 6. Анализ вклада в риск
    high_risk_assets = [
      symbol for symbol, contribution in risk_metrics.risk_contribution.items()
      if contribution > 1.5 / len(risk_metrics.risk_contribution)
    ]

    if high_risk_assets:
      recommendations['actions'].append(
        f"Уменьшить позиции в активах с высоким вкладом в риск: {', '.join(high_risk_assets)}"
      )

    return recommendations

  def _update_correlation_graph(self, corr_matrix: pd.DataFrame):
    """Обновляет граф корреляций для визуализации"""
    self.correlation_graph.clear()

    # Добавляем узлы
    for symbol in corr_matrix.index:
      self.correlation_graph.add_node(symbol)

    # Добавляем ребра с весами
    for i in range(len(corr_matrix)):
      for j in range(i + 1, len(corr_matrix)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.3:  # Только значимые корреляции
          self.correlation_graph.add_edge(
            corr_matrix.index[i],
            corr_matrix.columns[j],
            weight=corr,
            color='red' if corr > 0 else 'blue'
          )

  def _save_correlation_history(self, symbols: List[str], correlations: Dict):
    """Сохраняет историю корреляций для анализа изменений"""
    timestamp = datetime.now()

    for corr_type, corr_matrix in correlations.items():
      for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
          pair_key = f"{corr_matrix.index[i]}_{corr_matrix.columns[j]}"

          corr_data = CorrelationData(
            symbol1=corr_matrix.index[i],
            symbol2=corr_matrix.columns[j],
            correlation=corr_matrix.iloc[i, j],
            correlation_type=CorrelationType(corr_type),
            timeframe=Timeframe.ONE_HOUR,
            period_days=30,
            timestamp=timestamp
          )

          if pair_key not in self.correlation_history:
            self.correlation_history[pair_key] = []

          self.correlation_history[pair_key].append(corr_data)

          # Ограничиваем историю последними 100 записями
          if len(self.correlation_history[pair_key]) > 100:
            self.correlation_history[pair_key] = self.correlation_history[pair_key][-100:]

  async def adjust_position_sizes_by_correlation(self,
                                                 signals: Dict[str, Any],
                                                 current_positions: Dict[str, float]) -> Dict[str, float]:
    """
    Корректирует размеры позиций с учетом корреляций
    """
    if not self.correlation_matrix or self.correlation_matrix.empty:
      logger.warning("Нет данных о корреляциях для корректировки позиций")
      return {symbol: signal.get('size', 0) for symbol, signal in signals.items()}

    adjusted_sizes = {}

    for symbol, signal in signals.items():
      base_size = signal.get('size', 0)

      # Проверяем корреляции с существующими позициями
      correlation_penalty = 0

      for existing_symbol, existing_size in current_positions.items():
        if existing_symbol != symbol and existing_symbol in self.correlation_matrix.index:
          try:
            correlation = self.correlation_matrix.loc[symbol, existing_symbol]

            # Штраф за высокую корреляцию
            if abs(correlation) > self.correlation_threshold:
              penalty = (abs(correlation) - self.correlation_threshold) * existing_size
              correlation_penalty += penalty
              logger.info(
                f"Корреляция {symbol}-{existing_symbol}: {correlation:.2f}, "
                f"штраф: {penalty:.4f}"
              )
          except KeyError:
            continue

      # Корректируем размер с учетом штрафа
      adjustment_factor = max(0.3, 1 - correlation_penalty)
      adjusted_size = base_size * adjustment_factor

      adjusted_sizes[symbol] = adjusted_size

      if adjustment_factor < 1:
        logger.info(
          f"Размер позиции {symbol} скорректирован с {base_size:.4f} "
          f"до {adjusted_size:.4f} (фактор: {adjustment_factor:.2f})"
        )

    return adjusted_sizes

  def get_correlation_between(self, symbol1: str, symbol2: str) -> Optional[float]:
    """Получает корреляцию между двумя символами из кэша или матрицы"""
    # Проверяем кэш
    cache_key = f"{symbol1}_{symbol2}"
    if cache_key in self.correlation_cache:
      cached_data = self.correlation_cache[cache_key]
      if (datetime.now() - cached_data.timestamp).seconds < self.cache_ttl:
        return cached_data.correlation

    # Проверяем матрицу
    if self.correlation_matrix is not None:
      try:
        return self.correlation_matrix.loc[symbol1, symbol2]
      except KeyError:
        try:
          return self.correlation_matrix.loc[symbol2, symbol1]
        except KeyError:
          pass

    return None

  def get_least_correlated_assets(self, symbol: str,
                                  available_symbols: List[str],
                                  n_assets: int = 5) -> List[Tuple[str, float]]:
    """Находит наименее коррелированные активы с данным символом"""
    correlations = []

    for other_symbol in available_symbols:
      if other_symbol != symbol:
        corr = self.get_correlation_between(symbol, other_symbol)
        if corr is not None:
          correlations.append((other_symbol, abs(corr)))

    # Сортируем по возрастанию абсолютной корреляции
    correlations.sort(key=lambda x: x[1])

    return correlations[:n_assets]

  def should_block_signal_due_to_correlation(self, symbol: str,
                                             current_positions: List[str]) -> Tuple[bool, str]:
    """Проверяет, следует ли заблокировать сигнал из-за высокой корреляции"""
    if not current_positions:
      return False, ""

    for position_symbol in current_positions:
      corr = self.get_correlation_between(symbol, position_symbol)
      if corr is not None and abs(corr) > 0.95:
        return True, f"Экстремально высокая корреляция с {position_symbol} ({corr:.2f})"

    return False, ""

  def get_portfolio_correlation_summary(self, symbols: List[str]) -> Dict[str, Any]:
    """Получает сводку по корреляциям портфеля"""
    if not self.correlation_matrix or self.correlation_matrix.empty:
      return {"status": "no_data"}

    # Фильтруем матрицу для заданных символов
    available_symbols = [s for s in symbols if s in self.correlation_matrix.index]

    if len(available_symbols) < 2:
      return {"status": "insufficient_symbols"}

    sub_matrix = self.correlation_matrix.loc[available_symbols, available_symbols]

    # Вычисляем статистику
    upper_triangle = np.triu(sub_matrix, k=1)
    correlations = upper_triangle[upper_triangle != 0]

    return {
      'status': 'ok',
      'symbols': available_symbols,
      'avg_correlation': np.mean(np.abs(correlations)),
      'max_correlation': np.max(np.abs(correlations)),
      'min_correlation': np.min(correlations),
      'high_correlation_pairs': len(correlations[np.abs(correlations) > self.correlation_threshold]),
      'total_pairs': len(correlations),
      'last_update': self.correlation_cache.get('last_update', datetime.now())
    }