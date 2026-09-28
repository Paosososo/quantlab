# ADR-001: Every fact row carries `available_at` alongside `ts`

**Status.** Accepted, and load-bearing for the whole project.

## Context

Look-ahead bias is the failure mode that makes a quantitative research result
worthless, and it rarely arrives deliberately. It arrives through a line like
`df["close"].rolling(20).mean()` computed on the full history and joined back
onto a signal, or a macro series joined on its observation date rather than its
publication date.

Reviewing every line for this is unreliable. We wanted a control that does not
depend on remembering.

## Decision

Every fact table (`prices`, `returns`, `features`, `economic_data`) carries two
timestamps: `ts`, the instant the observation describes, and `available_at`, the
earliest instant it could have been known. Every read path that feeds a
prediction filters on `available_at <= decision_time`.

## Alternatives rejected

**Rely on index alignment and `shift(1)`.** The industry-standard shortcut. It
works until a series with a different publication schedule is joined in, at
which point a single `shift` is silently wrong by weeks. It also cannot express
"this daily series is available same-day, that monthly one is available six weeks
late".

**Store only `ts` and apply a global lag.** One lag cannot be right for both a
price bar and a CPI print.

**A separate `vintages` table.** The fully correct answer, and what ALFRED does.
Rejected as too heavy for the available data: the free endpoints do not expose
vintages, so the table would be populated by the same approximation we are
already applying, with more machinery.

## Consequences

*Good.* "Did I peek at the future?" becomes a SQL predicate. The database
enforces `CHECK (available_at >= ts)`, so a loader bug cannot create a row that
claims to have been knowable before the period it describes. The API can expose
`as_of` and answer "what could I have seen then?" for free.

*Costs.* Every table is one column wider. Every read path must remember to pass
`as_of` — the point-in-time view exists so that forgetting is caught, but a raw
`repo.load_prices()` without `as_of` still returns everything, which is right for
reporting and wrong for features.

*Approximations we accept.* Price availability uses the regular-hours close and
ignores early closes, making it a few hours late on a handful of days a year;
being late is harmless, being early is bias. Macro availability uses a
conservative per-series publication lag rather than a real release calendar.
