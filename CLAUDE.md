@AGENTS.md

<!-- The shared instructions live in AGENTS.md, which Codex and other agents read natively.
     Keep this import: Claude Code reads AGENTS.md on its own only when no CLAUDE.md or
     CLAUDE.local.md sits in the working directory or above it, and not at all on versions
     before 2.1.277, so without this file some sessions would load no project instructions.
     An import rather than a symlink because Windows checks symlinks out as plain text. -->
