"""Initial schema: reference data, market data, features, models, backtests

Revision ID: 0001
Revises: 
Create Date: 2026-09-04 07:34:04.815523+00:00
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import Text
from sqlalchemy.dialects import postgresql

revision: str = '0001'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('assets',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('symbol', sa.String(length=32), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=True),
    sa.Column('asset_class', sa.Enum('equity', 'etf', 'index', 'fx', 'commodity', 'rate', 'crypto', name='asset_class', native_enum=False, length=32), nullable=False),
    sa.Column('exchange', sa.String(length=32), nullable=True),
    sa.Column('currency', sa.String(length=3), nullable=False),
    sa.Column('session_name', sa.String(length=32), nullable=False),
    sa.Column('status', sa.Enum('active', 'delisted', 'suspended', name='asset_status', native_enum=False, length=32), nullable=False),
    sa.Column('listed_on', sa.Date(), nullable=True),
    sa.Column('delisted_on', sa.Date(), nullable=True),
    sa.Column('meta', sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), 'postgresql'), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.CheckConstraint('currency = upper(currency)', name=op.f('ck_assets_currency_upper')),
    sa.CheckConstraint('delisted_on IS NULL OR listed_on IS NULL OR delisted_on >= listed_on', name=op.f('ck_assets_delist_after_list')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_assets')),
    sa.UniqueConstraint('symbol', name='uq_assets_symbol')
    )
    with op.batch_alter_table('assets', schema=None) as batch_op:
        batch_op.create_index('ix_assets_asset_class_status', ['asset_class', 'status'], unique=False)

    op.create_table('economic_series',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('code', sa.String(length=64), nullable=False),
    sa.Column('title', sa.String(length=300), nullable=True),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('frequency', sa.Enum('daily', 'weekly', 'monthly', 'quarterly', 'annual', name='frequency', native_enum=False, length=32), nullable=False),
    sa.Column('units', sa.String(length=64), nullable=True),
    sa.Column('publication_lag_days', sa.Integer(), nullable=False),
    sa.Column('seasonal_adjustment', sa.String(length=32), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.CheckConstraint('publication_lag_days >= 0', name=op.f('ck_economic_series_non_negative_lag')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_economic_series')),
    sa.UniqueConstraint('code', name=op.f('uq_economic_series_code'))
    )
    op.create_table('feature_sets',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=100), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('spec_hash', sa.String(length=64), nullable=False),
    sa.Column('definition', sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), 'postgresql'), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_feature_sets')),
    sa.UniqueConstraint('name', 'version', name='uq_feature_sets_name_version')
    )
    op.create_table('ingestion_runs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('provider', sa.String(length=32), nullable=False),
    sa.Column('dataset', sa.String(length=64), nullable=False),
    sa.Column('entity', sa.String(length=64), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('status', sa.Enum('running', 'succeeded', 'failed', name='run_status', native_enum=False, length=32), nullable=False),
    sa.Column('rows_fetched', sa.Integer(), nullable=False),
    sa.Column('rows_written', sa.Integer(), nullable=False),
    sa.Column('rows_rejected', sa.Integer(), nullable=False),
    sa.Column('watermark_from', sa.DateTime(timezone=True), nullable=True),
    sa.Column('watermark_to', sa.DateTime(timezone=True), nullable=True),
    sa.Column('raw_path', sa.String(length=500), nullable=True),
    sa.Column('payload_sha256', sa.String(length=64), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_ingestion_runs'))
    )
    with op.batch_alter_table('ingestion_runs', schema=None) as batch_op:
        batch_op.create_index('ix_ingestion_runs_lookup', ['provider', 'dataset', 'entity', 'started_at'], unique=False)

    op.create_table('strategies',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('kind', sa.String(length=64), nullable=False),
    sa.Column('params', sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), 'postgresql'), nullable=False),
    sa.Column('params_hash', sa.String(length=64), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_strategies')),
    sa.UniqueConstraint('name', 'params_hash', name='uq_strategies_name_params')
    )
    op.create_table('universes',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('code', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_universes')),
    sa.UniqueConstraint('code', name=op.f('uq_universes_code'))
    )
    op.create_table('data_quality_checks',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('dataset', sa.String(length=64), nullable=False),
    sa.Column('entity', sa.String(length=64), nullable=True),
    sa.Column('check_name', sa.String(length=100), nullable=False),
    sa.Column('status', sa.Enum('passed', 'warned', 'failed', name='check_status', native_enum=False, length=32), nullable=False),
    sa.Column('observed', sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), 'postgresql'), nullable=True),
    sa.Column('message', sa.Text(), nullable=True),
    sa.Column('checked_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('ingestion_run_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['ingestion_run_id'], ['ingestion_runs.id'], name=op.f('fk_data_quality_checks_ingestion_run_id_ingestion_runs'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_data_quality_checks'))
    )
    with op.batch_alter_table('data_quality_checks', schema=None) as batch_op:
        batch_op.create_index('ix_data_quality_checks_dataset_checked_at', ['dataset', 'checked_at'], unique=False)
        batch_op.create_index('ix_data_quality_checks_status', ['status'], unique=False)

    op.create_table('economic_data',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('series_id', sa.Integer(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('available_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('value', sa.Float(), nullable=True),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('ingested_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint('available_at >= ts', name=op.f('ck_economic_data_available_after_ts')),
    sa.ForeignKeyConstraint(['series_id'], ['economic_series.id'], name=op.f('fk_economic_data_series_id_economic_series'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_economic_data')),
    sa.UniqueConstraint('series_id', 'ts', name='uq_economic_data_series_id_ts')
    )
    with op.batch_alter_table('economic_data', schema=None) as batch_op:
        batch_op.create_index('ix_economic_data_series_id_available_at', ['series_id', 'available_at'], unique=False)

    op.create_table('features',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('feature_set_id', sa.Integer(), nullable=False),
    sa.Column('asset_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=100), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('available_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('value', sa.Float(), nullable=True),
    sa.CheckConstraint('available_at >= ts', name=op.f('ck_features_available_after_ts')),
    sa.ForeignKeyConstraint(['asset_id'], ['assets.id'], name=op.f('fk_features_asset_id_assets'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['feature_set_id'], ['feature_sets.id'], name=op.f('fk_features_feature_set_id_feature_sets'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_features')),
    sa.UniqueConstraint('feature_set_id', 'asset_id', 'name', 'ts', name='uq_features_key')
    )
    with op.batch_alter_table('features', schema=None) as batch_op:
        batch_op.create_index('ix_features_lookup', ['feature_set_id', 'asset_id', 'ts'], unique=False)
        batch_op.create_index('ix_features_name_ts', ['name', 'ts'], unique=False)

    op.create_table('model_runs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('model_type', sa.String(length=64), nullable=False),
    sa.Column('feature_set_id', sa.Integer(), nullable=True),
    sa.Column('target_name', sa.String(length=100), nullable=False),
    sa.Column('config', sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), 'postgresql'), nullable=False),
    sa.Column('config_hash', sa.String(length=64), nullable=False),
    sa.Column('data_fingerprint', sa.String(length=64), nullable=True),
    sa.Column('code_version', sa.String(length=64), nullable=True),
    sa.Column('seed', sa.Integer(), nullable=False),
    sa.Column('validation_scheme', sa.String(length=64), nullable=False),
    sa.Column('train_start', sa.DateTime(timezone=True), nullable=True),
    sa.Column('train_end', sa.DateTime(timezone=True), nullable=True),
    sa.Column('test_start', sa.DateTime(timezone=True), nullable=True),
    sa.Column('test_end', sa.DateTime(timezone=True), nullable=True),
    sa.Column('status', sa.Enum('running', 'succeeded', 'failed', name='run_status', native_enum=False, length=32), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('metrics', sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), 'postgresql'), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.CheckConstraint('train_end IS NULL OR test_start IS NULL OR test_start >= train_end', name=op.f('ck_model_runs_test_after_train')),
    sa.ForeignKeyConstraint(['feature_set_id'], ['feature_sets.id'], name=op.f('fk_model_runs_feature_set_id_feature_sets'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_model_runs'))
    )
    with op.batch_alter_table('model_runs', schema=None) as batch_op:
        batch_op.create_index('ix_model_runs_config_hash', ['config_hash'], unique=False)
        batch_op.create_index('ix_model_runs_model_type_status', ['model_type', 'status'], unique=False)

    op.create_table('prices',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('asset_id', sa.Integer(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('available_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('open', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('high', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('low', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('close', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('adj_close', sa.Numeric(precision=20, scale=8), nullable=True),
    sa.Column('volume', sa.Numeric(precision=24, scale=10), nullable=True),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('ingested_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint('available_at >= ts', name=op.f('ck_prices_available_after_ts')),
    sa.CheckConstraint('high >= low', name=op.f('ck_prices_high_ge_low')),
    sa.CheckConstraint('high >= open AND high >= close', name=op.f('ck_prices_high_is_max')),
    sa.CheckConstraint('low <= open AND low <= close', name=op.f('ck_prices_low_is_min')),
    sa.CheckConstraint('open > 0 AND high > 0 AND low > 0 AND close > 0', name=op.f('ck_prices_positive_prices')),
    sa.CheckConstraint('volume IS NULL OR volume >= 0', name=op.f('ck_prices_volume_non_negative')),
    sa.ForeignKeyConstraint(['asset_id'], ['assets.id'], name=op.f('fk_prices_asset_id_assets'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_prices')),
    sa.UniqueConstraint('asset_id', 'ts', name='uq_prices_asset_id_ts')
    )
    with op.batch_alter_table('prices', schema=None) as batch_op:
        batch_op.create_index('ix_prices_asset_id_available_at', ['asset_id', 'available_at'], unique=False)
        batch_op.create_index('ix_prices_ts', ['ts'], unique=False)

    op.create_table('returns',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('asset_id', sa.Integer(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('available_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('horizon_days', sa.Integer(), nullable=False),
    sa.Column('direction', sa.Enum('trailing', 'forward', name='return_direction', native_enum=False, length=32), nullable=False),
    sa.Column('method', sa.Enum('simple', 'log', name='return_method', native_enum=False, length=32), nullable=False),
    sa.Column('value', sa.Float(), nullable=False),
    sa.Column('price_field', sa.String(length=16), nullable=False),
    sa.CheckConstraint('available_at >= ts', name=op.f('ck_returns_available_after_ts')),
    sa.CheckConstraint('horizon_days > 0', name=op.f('ck_returns_positive_horizon')),
    sa.ForeignKeyConstraint(['asset_id'], ['assets.id'], name=op.f('fk_returns_asset_id_assets'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_returns')),
    sa.UniqueConstraint('asset_id', 'ts', 'horizon_days', 'direction', 'method', 'price_field', name='uq_returns_key')
    )
    with op.batch_alter_table('returns', schema=None) as batch_op:
        batch_op.create_index('ix_returns_asset_id_ts', ['asset_id', 'ts'], unique=False)

    op.create_table('universe_members',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('universe_id', sa.Integer(), nullable=False),
    sa.Column('asset_id', sa.Integer(), nullable=False),
    sa.Column('valid_from', sa.DateTime(timezone=True), nullable=False),
    sa.Column('valid_to', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint('valid_to IS NULL OR valid_to > valid_from', name=op.f('ck_universe_members_interval_ordered')),
    sa.ForeignKeyConstraint(['asset_id'], ['assets.id'], name=op.f('fk_universe_members_asset_id_assets'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['universe_id'], ['universes.id'], name=op.f('fk_universe_members_universe_id_universes'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_universe_members')),
    sa.UniqueConstraint('universe_id', 'asset_id', 'valid_from', name='uq_universe_members_slice')
    )
    with op.batch_alter_table('universe_members', schema=None) as batch_op:
        batch_op.create_index('ix_universe_members_universe_id_valid_from', ['universe_id', 'valid_from'], unique=False)

    op.create_table('backtests',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('strategy_id', sa.Integer(), nullable=False),
    sa.Column('model_run_id', sa.Integer(), nullable=True),
    sa.Column('universe_id', sa.Integer(), nullable=True),
    sa.Column('benchmark_asset_id', sa.Integer(), nullable=True),
    sa.Column('start_ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('end_ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('initial_cash', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('execution_timing', sa.Enum('next_open', 'next_close', name='execution_timing', native_enum=False, length=32), nullable=False),
    sa.Column('cost_config', sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), 'postgresql'), nullable=False),
    sa.Column('engine_config', sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), 'postgresql'), nullable=False),
    sa.Column('config_hash', sa.String(length=64), nullable=False),
    sa.Column('code_version', sa.String(length=64), nullable=True),
    sa.Column('status', sa.Enum('running', 'succeeded', 'failed', name='run_status', native_enum=False, length=32), nullable=False),
    sa.Column('metrics', sa.JSON().with_variant(postgresql.JSONB(astext_type=Text()), 'postgresql'), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.CheckConstraint('end_ts >= start_ts', name=op.f('ck_backtests_end_after_start')),
    sa.CheckConstraint('initial_cash > 0', name=op.f('ck_backtests_positive_initial_cash')),
    sa.ForeignKeyConstraint(['benchmark_asset_id'], ['assets.id'], name=op.f('fk_backtests_benchmark_asset_id_assets'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['model_run_id'], ['model_runs.id'], name=op.f('fk_backtests_model_run_id_model_runs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['strategy_id'], ['strategies.id'], name=op.f('fk_backtests_strategy_id_strategies'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['universe_id'], ['universes.id'], name=op.f('fk_backtests_universe_id_universes'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_backtests'))
    )
    with op.batch_alter_table('backtests', schema=None) as batch_op:
        batch_op.create_index('ix_backtests_config_hash', ['config_hash'], unique=False)

    op.create_table('predictions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('model_run_id', sa.Integer(), nullable=False),
    sa.Column('asset_id', sa.Integer(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('target_ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('fold', sa.Integer(), nullable=False),
    sa.Column('split', sa.Enum('train', 'test', 'validation', name='split_kind', native_enum=False, length=32), nullable=False),
    sa.Column('y_pred', sa.Float(), nullable=False),
    sa.Column('y_true', sa.Float(), nullable=True),
    sa.Column('y_proba', sa.Float(), nullable=True),
    sa.CheckConstraint('target_ts > ts', name=op.f('ck_predictions_target_after_decision')),
    sa.ForeignKeyConstraint(['asset_id'], ['assets.id'], name=op.f('fk_predictions_asset_id_assets'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['model_run_id'], ['model_runs.id'], name=op.f('fk_predictions_model_run_id_model_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_predictions')),
    sa.UniqueConstraint('model_run_id', 'asset_id', 'ts', 'fold', name='uq_predictions_key')
    )
    with op.batch_alter_table('predictions', schema=None) as batch_op:
        batch_op.create_index('ix_predictions_run_ts', ['model_run_id', 'ts'], unique=False)

    op.create_table('orders',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('backtest_id', sa.Integer(), nullable=False),
    sa.Column('asset_id', sa.Integer(), nullable=False),
    sa.Column('client_order_id', sa.String(length=64), nullable=False),
    sa.Column('decision_ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('execution_ts', sa.DateTime(timezone=True), nullable=True),
    sa.Column('side', sa.Enum('buy', 'sell', name='order_side', native_enum=False, length=32), nullable=False),
    sa.Column('order_type', sa.Enum('market', 'limit', name='order_type', native_enum=False, length=32), nullable=False),
    sa.Column('quantity', sa.Numeric(precision=24, scale=10), nullable=False),
    sa.Column('limit_price', sa.Numeric(precision=20, scale=8), nullable=True),
    sa.Column('status', sa.Enum('pending', 'filled', 'partially_filled', 'rejected', 'cancelled', name='order_status', native_enum=False, length=32), nullable=False),
    sa.Column('reject_reason', sa.String(length=200), nullable=True),
    sa.CheckConstraint('execution_ts IS NULL OR execution_ts > decision_ts', name=op.f('ck_orders_execution_after_decision')),
    sa.CheckConstraint('quantity > 0', name=op.f('ck_orders_positive_quantity')),
    sa.ForeignKeyConstraint(['asset_id'], ['assets.id'], name=op.f('fk_orders_asset_id_assets'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['backtest_id'], ['backtests.id'], name=op.f('fk_orders_backtest_id_backtests'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_orders')),
    sa.UniqueConstraint('backtest_id', 'client_order_id', name='uq_orders_backtest_client_id')
    )
    with op.batch_alter_table('orders', schema=None) as batch_op:
        batch_op.create_index('ix_orders_backtest_decision_ts', ['backtest_id', 'decision_ts'], unique=False)

    op.create_table('portfolio_snapshots',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('backtest_id', sa.Integer(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('cash', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('positions_value', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('equity', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('gross_exposure', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('net_exposure', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('leverage', sa.Float(), nullable=False),
    sa.Column('n_positions', sa.Integer(), nullable=False),
    sa.Column('period_return', sa.Float(), nullable=True),
    sa.Column('drawdown', sa.Float(), nullable=True),
    sa.Column('benchmark_equity', sa.Numeric(precision=20, scale=8), nullable=True),
    sa.Column('turnover', sa.Float(), nullable=True),
    sa.ForeignKeyConstraint(['backtest_id'], ['backtests.id'], name=op.f('fk_portfolio_snapshots_backtest_id_backtests'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_portfolio_snapshots')),
    sa.UniqueConstraint('backtest_id', 'ts', name='uq_portfolio_snapshots_backtest_ts')
    )
    with op.batch_alter_table('portfolio_snapshots', schema=None) as batch_op:
        batch_op.create_index('ix_portfolio_snapshots_backtest_ts', ['backtest_id', 'ts'], unique=False)

    op.create_table('position_snapshots',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('backtest_id', sa.Integer(), nullable=False),
    sa.Column('asset_id', sa.Integer(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('quantity', sa.Numeric(precision=24, scale=10), nullable=False),
    sa.Column('price', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('market_value', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('weight', sa.Float(), nullable=True),
    sa.ForeignKeyConstraint(['asset_id'], ['assets.id'], name=op.f('fk_position_snapshots_asset_id_assets'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['backtest_id'], ['backtests.id'], name=op.f('fk_position_snapshots_backtest_id_backtests'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_position_snapshots')),
    sa.UniqueConstraint('backtest_id', 'asset_id', 'ts', name='uq_position_snapshots_key')
    )
    with op.batch_alter_table('position_snapshots', schema=None) as batch_op:
        batch_op.create_index('ix_position_snapshots_backtest_ts', ['backtest_id', 'ts'], unique=False)

    op.create_table('trades',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('backtest_id', sa.Integer(), nullable=False),
    sa.Column('order_id', sa.Integer(), nullable=True),
    sa.Column('asset_id', sa.Integer(), nullable=False),
    sa.Column('decision_ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('execution_ts', sa.DateTime(timezone=True), nullable=False),
    sa.Column('side', sa.Enum('buy', 'sell', name='order_side', native_enum=False, length=32), nullable=False),
    sa.Column('quantity', sa.Numeric(precision=24, scale=10), nullable=False),
    sa.Column('reference_price', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('fill_price', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('commission', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('slippage_cost', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('notional', sa.Numeric(precision=20, scale=8), nullable=False),
    sa.Column('realised_pnl', sa.Numeric(precision=20, scale=8), nullable=True),
    sa.CheckConstraint('commission >= 0 AND slippage_cost >= 0', name=op.f('ck_trades_non_negative_costs')),
    sa.CheckConstraint('execution_ts > decision_ts', name=op.f('ck_trades_execution_after_decision')),
    sa.CheckConstraint('fill_price > 0 AND reference_price > 0', name=op.f('ck_trades_positive_prices')),
    sa.CheckConstraint('quantity > 0', name=op.f('ck_trades_positive_quantity')),
    sa.ForeignKeyConstraint(['asset_id'], ['assets.id'], name=op.f('fk_trades_asset_id_assets'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['backtest_id'], ['backtests.id'], name=op.f('fk_trades_backtest_id_backtests'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['order_id'], ['orders.id'], name=op.f('fk_trades_order_id_orders'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_trades'))
    )
    with op.batch_alter_table('trades', schema=None) as batch_op:
        batch_op.create_index('ix_trades_backtest_execution_ts', ['backtest_id', 'execution_ts'], unique=False)



def downgrade() -> None:
    with op.batch_alter_table('trades', schema=None) as batch_op:
        batch_op.drop_index('ix_trades_backtest_execution_ts')

    op.drop_table('trades')
    with op.batch_alter_table('position_snapshots', schema=None) as batch_op:
        batch_op.drop_index('ix_position_snapshots_backtest_ts')

    op.drop_table('position_snapshots')
    with op.batch_alter_table('portfolio_snapshots', schema=None) as batch_op:
        batch_op.drop_index('ix_portfolio_snapshots_backtest_ts')

    op.drop_table('portfolio_snapshots')
    with op.batch_alter_table('orders', schema=None) as batch_op:
        batch_op.drop_index('ix_orders_backtest_decision_ts')

    op.drop_table('orders')
    with op.batch_alter_table('predictions', schema=None) as batch_op:
        batch_op.drop_index('ix_predictions_run_ts')

    op.drop_table('predictions')
    with op.batch_alter_table('backtests', schema=None) as batch_op:
        batch_op.drop_index('ix_backtests_config_hash')

    op.drop_table('backtests')
    with op.batch_alter_table('universe_members', schema=None) as batch_op:
        batch_op.drop_index('ix_universe_members_universe_id_valid_from')

    op.drop_table('universe_members')
    with op.batch_alter_table('returns', schema=None) as batch_op:
        batch_op.drop_index('ix_returns_asset_id_ts')

    op.drop_table('returns')
    with op.batch_alter_table('prices', schema=None) as batch_op:
        batch_op.drop_index('ix_prices_ts')
        batch_op.drop_index('ix_prices_asset_id_available_at')

    op.drop_table('prices')
    with op.batch_alter_table('model_runs', schema=None) as batch_op:
        batch_op.drop_index('ix_model_runs_model_type_status')
        batch_op.drop_index('ix_model_runs_config_hash')

    op.drop_table('model_runs')
    with op.batch_alter_table('features', schema=None) as batch_op:
        batch_op.drop_index('ix_features_name_ts')
        batch_op.drop_index('ix_features_lookup')

    op.drop_table('features')
    with op.batch_alter_table('economic_data', schema=None) as batch_op:
        batch_op.drop_index('ix_economic_data_series_id_available_at')

    op.drop_table('economic_data')
    with op.batch_alter_table('data_quality_checks', schema=None) as batch_op:
        batch_op.drop_index('ix_data_quality_checks_status')
        batch_op.drop_index('ix_data_quality_checks_dataset_checked_at')

    op.drop_table('data_quality_checks')
    op.drop_table('universes')
    op.drop_table('strategies')
    with op.batch_alter_table('ingestion_runs', schema=None) as batch_op:
        batch_op.drop_index('ix_ingestion_runs_lookup')

    op.drop_table('ingestion_runs')
    op.drop_table('feature_sets')
    op.drop_table('economic_series')
    with op.batch_alter_table('assets', schema=None) as batch_op:
        batch_op.drop_index('ix_assets_asset_class_status')

    op.drop_table('assets')
