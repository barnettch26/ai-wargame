from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4


class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3


class Heuristic(Enum):
    E0 = 0
    E1 = 1
    E2 = 2


##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount


##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = False
    max_turns: int | None = 100
    randomize_moves: bool = True
    broker: str | None = None
    heuristic: Heuristic | None = Heuristic.E0


##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    total_seconds: float = 0.0


##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall))

        # Create initial output file here
        self.create_output_file()

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_movement(self, coords: CoordPair) -> bool:
        """Validate a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        unit = self.get(coords.src)
        if unit is None or unit.player != self.next_player:
            return False

        target_coords = coords.dst
        all_adjacent_coords = coords.src.iter_adjacent()
        is_destination_adjacent = False

        for adjacent_coordinate in all_adjacent_coords:
            # Check if the target coordinates corresponds to one of the adjacent coordinates
            if adjacent_coordinate.row == target_coords.row and adjacent_coordinate.col == target_coords.col:
                is_destination_adjacent = True
                break

        # The source and destination must be adjacent
        if not is_destination_adjacent:
            return False

        in_combat = self.is_engaged_in_combat(coords)

        # AI, Firewall and Program units cannot move when engaged in combat (adjacent with adversarial unit)
        if (in_combat
                and (unit.type == UnitType.AI or unit.type == UnitType.Firewall or unit.type == UnitType.Program)):
            return False

        if unit.type == UnitType.AI or unit.type == UnitType.Firewall or unit.type == UnitType.Program:
            # Attacker's AI, Firewall and Program cannot move down or right
            if unit.player == Player.Attacker and (coords.dst.row > coords.src.row or coords.dst.col > coords.src.col):
                return False
            # Defender's AI, Firewall and Program cannot move up or left
            if unit.player == Player.Defender and (coords.dst.row < coords.src.row or coords.dst.col < coords.src.col):
                return False

        unit = self.get(coords.dst)
        return unit is None

    def is_engaged_in_combat(self, coords: CoordPair) -> bool:
        """Determines if the source unit is adjacent to any adversarial unit"""
        top = Coord(coords.src.row - 1, coords.src.col)
        bottom = Coord(coords.src.row + 1, coords.src.col)
        left = Coord(coords.src.row, coords.src.col - 1)
        right = Coord(coords.src.row, coords.src.col + 1)

        top_unit = self.get(top)
        bottom_unit = self.get(bottom)
        left_unit = self.get(left)
        right_unit = self.get(right)

        if (top_unit is not None and top_unit.player != self.next_player) \
                or (bottom_unit is not None and bottom_unit.player != self.next_player) \
                or (left_unit is not None and left_unit.player != self.next_player) \
                or (right_unit is not None and right_unit.player != self.next_player):
            return True

        return False

    def is_valid_attack(self, coords: CoordPair) -> bool:
        """An attack move is only valid if T is adjacent to S (up, down, left, right).
        T and S must be adversarial units"""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        unit = self.get(coords.src)
        if unit is None or unit.player != self.next_player:
            return False

        target_coords = coords.dst
        all_adjacent_coords = coords.src.iter_adjacent()

        for adjacent_coordinate in all_adjacent_coords:
            # Check if the target coordinates corresponds to one of the adjacent coordinates
            if adjacent_coordinate.row == target_coords.row and adjacent_coordinate.col == target_coords.col:
                target_unit = self.get(target_coords)
                source_unit = self.get(coords.src)

                if target_unit is None or source_unit is None:
                    continue

                # Check that T and S are adversarial units
                if source_unit.player != target_unit.player:
                    return True

        return False

    def is_valid_repair(self, coords: CoordPair) -> bool:
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        src_unit = self.get(coords.src)
        if src_unit is None or src_unit.player != self.next_player:
            return False

        adj_coords = coords.src.iter_adjacent()
        for adj_coord in adj_coords:
            if coords.dst.to_string() == adj_coord.to_string():
                target_unit = self.get(coords.dst)

                # Must be valid target that has taken dmg
                if target_unit is None or not self.next_player or target_unit.health == 9:
                    return False
                # Check if healable
                if src_unit.repair_amount(target_unit) == 0:
                    return False
                return True

        return False

    def is_valid_self_destruct(self, coords: CoordPair) -> bool:
        # Make sure not controlling other players units
        unit = self.get(coords.src)
        if unit is None or unit.player != self.next_player:
            return False
        # Self destruct only if src and dst are same
        if coords.src != coords.dst:
            return False
        return True

    def self_destruct_unit(self, coord: Coord):
        surrounding_coords = coord.iter_range(1)
        for target_coord in surrounding_coords:
            if target_coord == coord:
                # Instant kill self
                self.mod_health(target_coord, -9)
            # Will take care of None check for us
            self.mod_health(target_coord, -2)

    def perform_move(self, coords: CoordPair) -> Tuple[bool, str]:
        """Validate and perform a movement or attack or repair or self-destruct expressed as a CoordPair.
        TODO: WRITE MISSING CODE!!!"""

        if self.is_valid_movement(coords):
            unit_name = self.get(coords.src).to_string()
            self.set(coords.dst, self.get(coords.src))
            self.set(coords.src, None)

            return (True, "Moved {0} from {1} to {2}".format(unit_name, coords.src.to_string(), coords.dst.to_string()))

        if self.is_valid_attack(coords):
            source_unit = self.get(coords.src)
            target_unit = self.get(coords.dst)
            src_unit_name = source_unit.to_string()
            target_unit_name = target_unit.to_string()

            source_to_target_damage = source_unit.damage_amount(target_unit)
            target_to_source_damage = target_unit.damage_amount(source_unit)

            self.mod_health(coords.src, -target_to_source_damage)
            self.mod_health(coords.dst, -source_to_target_damage)

            return (True, "Unit {0} at coordinate {1} attacked unit {2} at coordinate {3}".format(src_unit_name,
                                                                                                  coords.src.to_string(),
                                                                                                  target_unit_name,
                                                                                                  coords.dst.to_string()))

        if self.is_valid_self_destruct(coords):
            unit_name = self.get(coords.src).to_string()
            self.self_destruct_unit(coords.src)
            return (True, "Unit {0} at coordinate {1} self-destructed".format(unit_name, coords.src))

        if self.is_valid_repair(coords):
            src_unit_name = self.get(coords.src).to_string()
            target_unit_name = self.get(coords.dst).to_string()
            src_unit = self.get(coords.src)
            target_unit = self.get(coords.dst)

            self.mod_health(coords.dst, +src_unit.repair_amount(target_unit))
            return (True, "Unit {0} at coordinate {1} repaired unit {2} at coordinate {3}".format(src_unit_name,
                                                                                                  coords.src.to_string(),
                                                                                                  target_unit_name,
                                                                                                  coords.dst.to_string()))

        return (False, "invalid move")

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        output += self.board_to_string()
        return output

    def board_to_string(self) -> str:
        dim = self.options.dim
        coord = Coord()
        output = "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ", end='')
                    print(result)
                    if success:
                        self.update_output_file_turn(result)
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success, result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ", end='')
                    print(result)
                    self.update_output_file_turn(result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.find_best_computer_move()
        if mv is not None:
            (success, result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ", end='')
                print(result)
                self.update_output_file_turn(result)
                # TODO: print AI specific stuff
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord, unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        elif self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        elif self._defender_has_ai:
            return Player.Defender
        # In the situation where an both AIs kill each other, defender should win according to rules since
        # according to End of game section, A player wins if their AI is alive while other AI is destroyed, else defender wins.
        # This situation falls in the else
        return Player.Defender

    def move_candidates(self, player: Player) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src, _) in self.player_units(player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if (self.is_valid_movement(move) or
                        self.is_valid_attack(move) or
                        self.is_valid_repair(move) or
                        self.is_valid_self_destruct(move)):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def find_best_computer_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
        start_time = datetime.now()

        if self.options.alpha_beta:
            (score, move) = self.minimax_alpha_beta(self, 0, float('-inf'), float('inf'), self.next_player == Player.Attacker, start_time)
        else:
            (score, move) = self.minimax(self, 0, self.next_player == Player.Attacker, start_time)

        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        print(f"Evals per depth: ", end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ", end='')
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move

    def minimax(self, game: Game, depth: int, is_maximizing_player: bool, start_time: datetime) -> Tuple[
        float, CoordPair | None]:
        if depth == game.options.max_depth or game.has_winner() or (
                (datetime.now() - start_time).seconds >= game.options.max_time):
            if game.options.heuristic == Heuristic.E0:
                return game.calculate_heuristic_e0(), None
            elif game.options.heuristic == Heuristic.E1:
                return game.calculate_heuristic_e1(), None

        if is_maximizing_player:
            max_score = MIN_HEURISTIC_SCORE
            best_move = None

            for move in list(game.move_candidates(Player.Attacker)):
                game_copy = game.clone()
                game_copy.perform_move(move)

                score = self.minimax(game_copy, depth + 1, False, start_time)[0]

                max_score = max(score, max_score)
                if max_score == score:
                    best_move = move

                if (datetime.now() - start_time).seconds >= game.options.max_time:
                    break

            return max_score, best_move

        else:
            min_score = MAX_HEURISTIC_SCORE
            best_move = None

            for move in list(game.move_candidates(Player.Defender)):
                game_copy = game.clone()
                game_copy.perform_move(move)

                score = self.minimax(game_copy, depth + 1, True, start_time)[0]

                min_score = min(score, min_score)
                if min_score == score:
                    best_move = move

                if (datetime.now() - start_time).seconds >= game.options.max_time:
                    break

            return min_score, best_move

    # Minimax with alpha beta pruning.
    def minimax_alpha_beta(self, game: Game, depth: int, alpha: float, beta: float, is_maximizing_player: bool,
                           start_time: datetime) -> Tuple[float, CoordPair | None]:
        if depth == game.options.max_depth or game.has_winner() or (
                (datetime.now() - start_time).seconds >= game.options.max_time):
            if game.options.heuristic == Heuristic.E0:
                return game.calculate_heuristic_e0(), None
            elif game.options.heuristic == Heuristic.E1:
                return game.calculate_heuristic_e1(), None

        if is_maximizing_player:
            max_score = MIN_HEURISTIC_SCORE
            best_move = None

            for move in list(game.move_candidates(Player.Attacker)):
                game_copy = game.clone()
                game_copy.perform_move(move)

                score = self.minimax_alpha_beta(game_copy, depth + 1, alpha, beta, False, start_time)[0]
                max_score = max(max_score, score)

                alpha = max(alpha, score)
                if beta <= alpha:
                    break

                if max_score == score:
                    best_move = move

                if (datetime.now() - start_time).seconds >= game.options.max_time:
                    break

            return max_score, best_move

        else:
            min_score = MAX_HEURISTIC_SCORE
            best_move = None

            for move in list(game.move_candidates(Player.Defender)):
                game_copy = game.clone()
                game_copy.perform_move(move)

                score = self.minimax_alpha_beta(game_copy, depth + 1, alpha, beta, True, start_time)[0]
                min_score = min(min_score, score)

                beta = min(beta, score)
                if beta <= alpha:
                    break

                if min_score == score:
                    best_move = move

                if (datetime.now() - start_time).seconds >= game.options.max_time:
                    break

            return min_score, best_move

    def calculate_heuristic_e0(self) -> float:
        """e0 = (3VP1 + 3TP1 + 3FP1 + 3PP1 + 9999AIP1) − (3VP2 + 3TP2 + 3FP2 + 3PP2 + 9999AIP2)

        where:

        VPi = nb of Virus of Player i
        TPi = nb of Tech of Player i
        FPi = nb of Firewall of Player i
        PPi = nb of Program of Player i
        AIPi = nb of AI of Player i
        """

        nb_attacker_virus = 0
        nb_attacker_tech = 0
        nb_attacker_firewall = 0
        nb_attacker_program = 0
        nb_attacker_ai = 0

        nb_defender_virus = 0
        nb_defender_tech = 0
        nb_defender_firewall = 0
        nb_defender_program = 0
        nb_defender_ai = 0

        board_dimension = self.options.dim

        for row in range(board_dimension):
            for col in range(board_dimension):
                coord = Coord(row, col)

                unit = self.get(coord)

                if unit is not None:
                    if unit.player == Player.Attacker and unit.type == UnitType.Virus:
                        nb_attacker_virus += 1
                    elif unit.player == Player.Attacker and unit.type == UnitType.Tech:
                        nb_attacker_tech += 1
                    elif unit.player == Player.Attacker and unit.type == UnitType.Firewall:
                        nb_attacker_firewall += 1
                    elif unit.player == Player.Attacker and unit.type == UnitType.Program:
                        nb_attacker_program += 1
                    elif unit.player == Player.Attacker and unit.type == UnitType.AI:
                        nb_attacker_ai += 1

                    elif unit.player == Player.Defender and unit.type == UnitType.Virus:
                        nb_defender_virus += 1
                    elif unit.player == Player.Defender and unit.type == UnitType.Tech:
                        nb_defender_tech += 1
                    elif unit.player == Player.Defender and unit.type == UnitType.Firewall:
                        nb_defender_firewall += 1
                    elif unit.player == Player.Defender and unit.type == UnitType.Program:
                        nb_defender_program += 1
                    elif unit.player == Player.Defender and unit.type == UnitType.AI:
                        nb_defender_ai += 1

        attacker_heuristic_value = (3 * nb_attacker_virus + 3 * nb_attacker_tech + 3 * nb_attacker_firewall +
                                    3 * nb_attacker_program + 9999 * nb_attacker_ai)

        defender_heuristic_value = (3 * nb_defender_virus + 3 * nb_defender_tech + 3 * nb_defender_firewall +
                                    3 * nb_defender_program + 9999 * nb_defender_ai)

        heuristic_value = attacker_heuristic_value - defender_heuristic_value

        return heuristic_value

    def calculate_heuristic_e1(self) -> float:
        total_health_attacker_virus = 0
        total_health_attacker_firewall = 0
        total_health_attacker_program = 0
        total_health_attacker_ai = 0

        total_health_defender_tech = 0
        total_health_defender_firewall = 0
        total_health_defender_program = 0
        total_health_defender_ai = 0

        board_dimension = self.options.dim

        for row in range(board_dimension):
            for col in range(board_dimension):
                coord = Coord(row, col)

                unit = self.get(coord)

                if unit is not None:
                    # Attacker units
                    if unit.player == Player.Attacker and unit.type == UnitType.Virus:
                        total_health_attacker_virus += unit.health
                    elif unit.player == Player.Attacker and unit.type == UnitType.Firewall:
                        total_health_attacker_firewall += unit.health
                    elif unit.player == Player.Attacker and unit.type == UnitType.Program:
                        total_health_attacker_program += unit.health
                    elif unit.player == Player.Attacker and unit.type == UnitType.AI:
                        total_health_attacker_ai += unit.health

                    # Defender units
                    elif unit.player == Player.Defender and unit.type == UnitType.Tech:
                        total_health_defender_tech += unit.health
                    elif unit.player == Player.Defender and unit.type == UnitType.Firewall:
                        total_health_defender_firewall += unit.health
                    elif unit.player == Player.Defender and unit.type == UnitType.Program:
                        total_health_defender_program += unit.health
                    elif unit.player == Player.Defender and unit.type == UnitType.AI:
                        total_health_defender_ai += unit.health

        # TODO: Add number of viruses adjacent to defender's AI
        attacker_heuristic_value = (100 * total_health_attacker_virus + 2 * total_health_attacker_program
                                    + 2 * total_health_attacker_firewall + 999 * total_health_attacker_ai)

        defender_heuristic_value = (5 * total_health_defender_tech + 5 * total_health_defender_firewall
                                    + 2 * total_health_defender_program + 9999 * total_health_defender_ai)

        heuristic_value = attacker_heuristic_value - defender_heuristic_value

        return heuristic_value

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played + 1:
                        move = CoordPair(
                            Coord(data['from']['row'], data['from']['col']),
                            Coord(data['to']['row'], data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

    def create_output_file(self):
        file = open(
            'gameTrace-{0}-{1}-{2}.txt'.format(self.options.alpha_beta, self.options.max_time, self.options.max_turns),
            "w")
        file.write("1. Game Parameters\n")
        file.write("    a) Timeout = {0}\n".format(self.options.max_time))
        file.write("    b) Max Turns: {0}\n".format(self.options.max_turns))
        if self.options.game_type == GameType.AttackerVsComp:
            file.write("    c) Alpha-Beta: {0}\n".format(self.options.alpha_beta))
            file.write("    d) Play Mode: Attacker = Human | Defender = AI\n")
            file.write("    e) Heuristic: {0}\n".format(self.options.heuristic.name))
        elif self.options.game_type == GameType.CompVsDefender:
            file.write("    c) Alpha-Beta: {0}\n".format(self.options.alpha_beta))
            file.write("    d) Play Mode: Attacker = AI | Defender = Human\n")
            file.write("    e) Heuristic: {0}\n".format(self.options.heuristic.name))
        elif self.options.game_type == GameType.CompVsComp:
            file.write("    c) Alpha-Beta: {0}\n".format(self.options.alpha_beta))
            file.write("    d) Play Mode: Attacker = AI | Defender = AI\n")
            file.write("    e) Heuristic: {0}\n".format(self.options.heuristic.name))
        else:
            file.write("    c) Alpha-Beta: No AI in this game.\n")
            file.write("    d) Play Mode: Attacker = Human | Defender = Human\n")
            file.write("    e) Heuristic: No AI in this game.\n")
        file.write("\n2. Initial Configuration of board:\n")
        file.write(self.board_to_string())

    def update_output_file_turn(self, move_string):
        file = open(
            'gameTrace-{0}-{1}-{2}.txt'.format(self.options.alpha_beta, self.options.max_time, self.options.max_turns),
            "a")
        file.write("\n\n\n")
        file.write("Turn Number: {0}\n".format(self.turns_played + 1))
        file.write("Player: {0}\n".format(self.next_player.name))
        file.write("Action: {0}\n".format(move_string))

        # TODO: More outputs for AI information and cumulative statistics

        file.write(self.board_to_string())

    def update_output_file_end(self, winner):
        file = open(
            'gameTrace-{0}-{1}-{2}.txt'.format(self.options.alpha_beta, self.options.max_time, self.options.max_turns),
            "a")
        file.write("\n{0} wins in {1} turns".format(winner, self.turns_played))


##############################################################################################################

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    parser.add_argument('--max_turns', type=int, help='maximum number of turns allowed')
    parser.add_argument('--alpha_beta', type=bool, action=argparse.BooleanOptionalAction,
                        help='Toggle Alpha-Beta for AI')
    parser.add_argument('--heuristic', type=str, default="e0", help="heuristic that an AI should use: e0 | e1 | e2")

    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    heuristic = None

    if args.heuristic == "e0":
        heuristic = Heuristic.E0
    elif args.heuristic == "e1":
        heuristic = Heuristic.E1
    elif args.heuristic == "e2":
        heuristic = Heuristic.E2

    # set up game options
    options = Options(game_type=game_type, heuristic=heuristic)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker
    if args.max_turns is not None:
        options.max_turns = args.max_turns
    if args.alpha_beta is not None:
        options.alpha_beta = args.alpha_beta

    # create a new game
    game = Game(options=options)

    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!")
            game.update_output_file_end(winner.name)
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)


##############################################################################################################

if __name__ == '__main__':
    main()
