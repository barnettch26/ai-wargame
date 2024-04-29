Team Name: JSF Devs

Team Members:
- Barnett Chengberlin
- Daniel Chin
- Stephen He

---

### The program can be used with multiple command line arguments:

- `--max_depth` : The maximum search depth that the minimax or alpha-beta pruning algorithms will reach (ex: 5)
- `--max_time` : The maximum search time before the algorithm stops and returns the best value found (ex: 5)
- `--game_type` :
	- `auto` -> AI vs AI
	- `attacker` -> Human vs AI
	- `defender` -> AI vs Human
	- `manual` -> Human vs Human
- `--max_turns` : The maximum number of turns allows during an entire game (ex: 100)
- `--alpha_beta` : Whether or not alpha-beta pruning will be used for the minimax algorithm (ex: simply write `--alpha_beta`)
- `--heuristic` : The heuristic that the AI will use (ex: `--heuristic e0` | `--heuristic e1` | `--heuristic e2`)

---

### Sample commands for Human turn:

#### Movement:
`[Source][Destination]`

Example input: `A1A2`

#### Attack:
`[Source][Destination]`

Example input: `A1A2`

#### Repair:
`[Source][Destination]`

Example input: `A1A2`

#### Self-Destruct:
`[Source][Source]`

Example input: `A1A1`
