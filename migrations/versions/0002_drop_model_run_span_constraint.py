"""Drop the model_runs test-after-train constraint

The original schema asserted ``test_start >= train_end`` on ``model_runs``.  That
is correct for a single holdout split and wrong for walk-forward validation,
where the aggregate spans overlap by construction: fold 3 trains on data that
fold 1 already tested on.  An integration test caught it when the first
multi-fold run failed to persist.

The per-fold ordering guarantee is unchanged and is enforced where it is
actually true: ``quantlab.features.guards.assert_split_ordering`` raises if any
fold's training data reaches into its test window, and ``predictions`` still
carries ``CHECK (target_ts > ts)``.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Batch mode so the migration also applies on SQLite, which cannot drop a
    # CHECK constraint in place and has to rebuild the table.
    with op.batch_alter_table("model_runs") as batch:
        # The bare name: MetaData's naming convention supplies the
        # "ck_model_runs_" prefix, and passing the full name would have it
        # applied twice.
        batch.drop_constraint("test_after_train", type_="check")


def downgrade() -> None:
    with op.batch_alter_table("model_runs") as batch:
        batch.create_check_constraint(
            "test_after_train",
            "train_end IS NULL OR test_start IS NULL OR test_start >= train_end",
        )
