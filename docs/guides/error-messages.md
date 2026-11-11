# Error Messages

Eden's design rule "fail fast, fail loud" decides *when* to raise: at the earliest well-defined point (Pydantic
validation, a term's `configure()` or `build()`), never deep inside a hot `compute()` step. This page decides *what
the error says*. It is for anyone writing a `raise` in Eden, and it assumes the reader of the message is often a
coding agent working from a captured traceback, with no running process, no debugger, and no memory of the session
that produced the log.

That reader has a narrow loop: grep for a symbol, open one or two files, make an edit. A message earns its keep by
removing lookups from that loop. A message that describes the problem well but hands over no identifier costs
*more* than a short one, because the reader still has to find the identifier and now has prose to discount first.

## The shape

```
<what broke>: '<the offending value>'.
<observed state — what was actually there, enumerated>
Fix: <an expression or edit the reader can type>
<the wrong fix, named, when a plausible wrong fix exists>
```

Not every message needs all four lines. Every message needs the first and the third.

A worked example, from `CurriculumManager.get_state` (`eden/managers/curriculum_manager.py`), when a curriculum
term reads another that is declared after it:

```
Curriculum term 'ep_tracker' has not computed yet, so it has no state to read.
curriculum_options order: penalty_torques, ep_tracker
'penalty_torques' is reading 'ep_tracker', but 'ep_tracker' is declared after it
Fix: move 'ep_tracker' above 'penalty_torques' in curriculum_options — CurriculumManager computes terms in declaration order, so a producer declared after its consumer would feed it a value one control step stale.
```

The reader needs no second lookup: the two config keys, their order, which one is at fault, and the edit are all
there.

## Rules

**Quote the offending value, first.** `'ep_trackr'` in quotes is a grep target; "an unknown term" is not. This
extends Eden's base rule that a message names the offending value and the available options.

**Enumerate what *is* there.** `Declared curriculum terms: level, penalty` turns a typo into a one-character edit.
The code that raises usually knows the roster, so make it pay. A registry lookup does this for you:
`Registry.get` raises a `KeyError` that names the missing key, suggests the closest registered key
(`Did you mean '...'?`) and lists the available ones.

**Give the fix as something typeable.** `curriculum_manager.get_term('level')` beats "read the level off the term".
An expression is copy-pasteable and unambiguous; a description has to be translated, and that translation is where
a reader guesses. A command counts too: `eden terms --grep <text>` is a fix a reader can run.

**Resolve the symbol for the reader.** A traceback carries the *class* that raised; it does not carry the *config
key* that built it, and a task can hold a dozen entries of one class (`metric_reward` twelve times in robosuite
PickPlace, two `PenaltyCurriculum`s sharing a tracker). Name the entry to edit *and* a class name to open. Eden
already does this at two layers, so build on them rather than repeating them:

- `build_term` in `eden/managers/base.py`, which every manager's `_build_term` goes through, adds a note
  (`add_note`) naming the config entry and its registry key to any exception a term raises while it is constructed
  or built: `while building RewardManager term 'lift_milk' (name='...')`. A term's own message therefore does not
  need to restate its config key.
- A term bound to an entity opens its errors with `term_subject(self)` (from `eden.utils.string`), which renders
  `"<Class> on entity '<name>'"`, rather than spelling the class or the entity out by hand.
  `CurriculumManager._term_label` does the same inline for curriculum terms (the class name, or `func()` for a
  function term).

**Name the wrong fix when there is an attractive one.** A reader told to make an error go away will reach for a
`try`/`except` or a default value. If that would reintroduce the bug the raise exists to prevent, say so in the
message. Two examples from the code:

- `CurriculumManager.get_state` at eval: "Do NOT catch this and fall back to a default: the default would stand for
  every step of the eval, grading it at a difficulty no training step used, with nothing logged and nothing raised."
- A term's options built while `eden.managers` is still importing (`eden/options/managers/base.py`): "Do not wrap
  this in try/except and do not make it fall back to ..." — `tests/options/managers/test_base.py` checks that the
  message keeps that line.

This is the single highest-value line in Eden's error text: it is the only place the wrong fix can be pre-empted at
the moment someone is considering it.

**Split conflated causes.** Two causes with one message force the reader to work out which applies, and sometimes
they pick wrong and edit the wrong file. If the code can tell the causes apart, it must. `CurriculumManager` records
the term it is in the middle of computing for exactly this reason, so "declared after its consumer" and "read before
the run's first `compute()`" — identical empty-cache states — get different text and different fixes.

**Point at something that ships.** A docs path (`docs/guides/environments.md`) or a CLI command
(`eden terms --grep <name>`) is readable offline, from the checkout or the installed package. An issue number is not.
Issue numbers belong in comments and docstrings, where they record *why* a rule exists, not in runtime messages,
where the reader needs *what to do*.

**Never put a newline in a `KeyError`.** `KeyError.__str__` is `repr(args[0])`, so a multi-line message reaches the
reader as one escaped, `\n`-laden blob — worse than the single line it was trying to improve on. Keep a `KeyError`
to one line, and use `ValueError`, `RuntimeError` or `TypeError` for multi-line text. (`Registry.get` keeps its
`KeyError` to one sentence on one line for this reason.)

**Message-building code must not be able to raise.** A helper that formats a label and throws `AttributeError`
replaces the diagnostic with a traceback *into the diagnostic*, and the original failure is lost. Use `isinstance`
over duck typing and `getattr(..., default)` over attribute access, and prefer a degraded label to an exception.
`CurriculumManager._term_label` is written this way, and `Registry.get` computes its "did you mean" suggestion only
when the missing key is a string, because a `None` key would make `difflib` raise from inside the message.

**Render it before you ship it.** Nearly every rule above was written after looking at real output rather than at
the source. Print the message from a script that triggers it, and read it as the person who will see it. It takes a
minute, and it is where the escaped `KeyError` and the raising label helper were both caught.

## What this is not

Not a request for longer messages. Every line above carries an identifier or an instruction. Filler ("something went
wrong", "please check your configuration", a restatement of the exception type) makes a message worse in exactly the
way prose does. Shorter is better *at equal information*.

Not a structured error framework either. Eden raises plain built-in exceptions with hand-written text, tuned per
site, and the exception type is part of the contract that callers and tests match on. There is no error-code
registry to look values up in.

## See also

- [Terms guide](terms.md) — where validation belongs in a term's lifecycle
- [Testing](testing.md) — asserting on a raised error and its message
- [Contributing](../contributing/index.md)
