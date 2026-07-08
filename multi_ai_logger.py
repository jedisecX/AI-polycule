#!/usr/bin/env python3

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class StateVector:
    labels: List[str]
    amplitudes: np.ndarray

    def normalize(self) -> "StateVector":
        norm = np.linalg.norm(self.amplitudes)
        if norm == 0:
            raise ValueError("State vector cannot have zero norm")
        self.amplitudes = self.amplitudes / norm
        return self

    def probabilities(self) -> Dict[str, float]:
        probabilities = np.abs(self.amplitudes) ** 2
        return {
            label: float(probability)
            for label, probability in zip(self.labels, probabilities)
        }


class QuantumStateEngine:
    """
    Experimental quantum-inspired hypothesis-state engine.
    This is NOT a physical quantum computer simulation.

    H(t) = D(t) + lambda * C(t)
    |psi(t + dt)> = exp(-i H dt) |psi(t)>

    Natural units: hbar = 1.
    """

    def __init__(self, labels: List[str]):
        if not labels:
            raise ValueError("At least one hypothesis label is required")
        if len(labels) != len(set(labels)):
            raise ValueError("Hypothesis labels must be unique")

        self.labels = labels
        n = len(labels)
        initial_amplitude = 1.0 / np.sqrt(n)
        self.state = StateVector(
            labels=labels,
            amplitudes=np.full(n, initial_amplitude, dtype=np.complex128),
        )

    def build_hamiltonian(
        self,
        hypothesis_potentials: np.ndarray,
        coupling: np.ndarray,
        lambda_val: float = 0.7,
    ) -> np.ndarray:
        hypothesis_potentials = np.asarray(
            hypothesis_potentials, dtype=np.float64
        )
        coupling = np.asarray(coupling, dtype=np.complex128)
        n = len(self.labels)

        if hypothesis_potentials.shape != (n,):
            raise ValueError(
                f"Expected {n} hypothesis potentials; "
                f"got {hypothesis_potentials.shape}"
            )
        if coupling.shape != (n, n):
            raise ValueError(
                f"Coupling matrix must have shape {(n, n)}; got {coupling.shape}"
            )
        if not np.all(np.isfinite(hypothesis_potentials)):
            raise ValueError("Hypothesis potentials must be finite")
        if not np.all(np.isfinite(coupling)):
            raise ValueError("Coupling matrix must be finite")
        if not np.isfinite(lambda_val):
            raise ValueError("lambda_val must be finite")
        if not np.allclose(coupling, coupling.conj().T, atol=1e-12):
            raise ValueError("Coupling matrix must be Hermitian")

        diagonal = np.diag(
            hypothesis_potentials.astype(np.complex128)
        )
        hamiltonian = diagonal + lambda_val * coupling

        if not np.allclose(
            hamiltonian, hamiltonian.conj().T, atol=1e-12
        ):
            raise RuntimeError("Constructed Hamiltonian is not Hermitian")

        return hamiltonian

    def evolve(self, hamiltonian: np.ndarray, dt: float = 1.0) -> None:
        hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
        n = len(self.labels)

        if hamiltonian.shape != (n, n):
            raise ValueError(f"Hamiltonian must have shape {(n, n)}")
        if not np.isfinite(dt):
            raise ValueError("dt must be finite")
        if not np.allclose(
            hamiltonian, hamiltonian.conj().T, atol=1e-12
        ):
            raise ValueError("Hamiltonian must be Hermitian")

        norm_before = np.linalg.norm(self.state.amplitudes)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        phases = np.exp(-1j * eigenvalues * dt)
        evolution_operator = (
            eigenvectors
            @ np.diag(phases)
            @ eigenvectors.conj().T
        )
        self.state.amplitudes = (
            evolution_operator @ self.state.amplitudes
        )
        norm_after = np.linalg.norm(self.state.amplitudes)

        if not np.isclose(norm_before, norm_after, atol=1e-10):
            raise RuntimeError(
                "State norm changed during unitary evolution: "
                f"{norm_before} -> {norm_after}"
            )

        self.state.normalize()

    def observe(self) -> Dict[str, float]:
        return self.state.probabilities()

    def most_likely_state(self) -> str:
        probabilities = self.observe()
        return max(probabilities, key=probabilities.get)


class ClassicalConsensusEngine:
    def evaluate(
        self, labels: List[str], assessments: np.ndarray
    ) -> Dict[str, float]:
        assessments = np.asarray(assessments, dtype=np.float64)

        if assessments.ndim != 2:
            raise ValueError("Assessments must be a 2D matrix")
        if assessments.shape[1] != len(labels):
            raise ValueError(
                "Assessment column count must match hypothesis label count"
            )
        if not np.all(np.isfinite(assessments)):
            raise ValueError("Assessments must contain finite values")
        if np.any(assessments < 0):
            raise ValueError("Assessments cannot be negative")

        row_sums = assessments.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0):
            raise ValueError("Assessment rows cannot sum to zero")

        normalized = assessments / row_sums
        consensus = normalized.mean(axis=0)
        consensus = consensus / consensus.sum()

        return {
            label: float(probability)
            for label, probability in zip(labels, consensus)
        }


def evidence_to_potential(
    support: float, oppose: float, uncertainty: float
) -> float:
    values = (support, oppose, uncertainty)

    if not all(np.isfinite(value) for value in values):
        raise ValueError("Evidence values must be finite")
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("Evidence values must be in [0, 1]")

    net_support = support - oppose
    confidence = 1.0 - uncertainty
    score = net_support * confidence
    return -float(score)


def disagreement_coupling(assessments: np.ndarray) -> np.ndarray:
    assessments = np.asarray(assessments, dtype=np.float64)

    if assessments.ndim != 2:
        raise ValueError("Assessments must be a 2D matrix")
    if not np.all(np.isfinite(assessments)):
        raise ValueError("Assessments must contain finite values")
    if np.any(assessments < 0):
        raise ValueError("Assessments cannot be negative")

    _, hypothesis_count = assessments.shape
    coupling = np.zeros(
        (hypothesis_count, hypothesis_count), dtype=np.float64
    )

    for i in range(hypothesis_count):
        for j in range(i + 1, hypothesis_count):
            delta = np.abs(
                assessments[:, i] - assessments[:, j]
            )
            disagreement = float(np.mean(delta))
            coupling[i, j] = disagreement
            coupling[j, i] = disagreement

    return coupling


class MultiAILogger:
    def __init__(self, db_path: str = "multi_ai_incidents.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA busy_timeout = 5000;")
        self._create_tables()
        self.quantum: Optional[QuantumStateEngine] = None
        self.classical = ClassicalConsensusEngine()

    def _create_tables(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            session_uuid TEXT NOT NULL UNIQUE,
            title TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS participants (
            id INTEGER PRIMARY KEY,
            participant_key TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            participant_type TEXT NOT NULL
                CHECK (participant_type IN ('human', 'model', 'system')),
            provider TEXT,
            model_name TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            event_uuid TEXT NOT NULL UNIQUE,
            session_id INTEGER NOT NULL,
            participant_id INTEGER NOT NULL,
            reply_to_event_id INTEGER,
            sequence_no INTEGER NOT NULL,
            event_type TEXT NOT NULL DEFAULT 'message',
            content TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (participant_id) REFERENCES participants(id),
            FOREIGN KEY (reply_to_event_id) REFERENCES events(id),
            UNIQUE (session_id, sequence_no)
        );

        CREATE TABLE IF NOT EXISTS quantum_states (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL,
            state_uuid TEXT NOT NULL UNIQUE,
            labels_json TEXT NOT NULL,
            amplitudes_real_json TEXT NOT NULL,
            amplitudes_imag_json TEXT NOT NULL,
            probabilities_json TEXT NOT NULL,
            most_likely_state TEXT NOT NULL,
            evolution_step INTEGER NOT NULL,
            lambda_value REAL NOT NULL,
            dt_value REAL NOT NULL,
            hamiltonian_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            UNIQUE (session_id, evolution_step)
        );

        CREATE TABLE IF NOT EXISTS consensus_snapshots (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL,
            snapshot_uuid TEXT NOT NULL UNIQUE,
            engine_type TEXT NOT NULL
                CHECK (engine_type IN ('classical', 'quantum_inspired')),
            probabilities_json TEXT NOT NULL,
            most_likely_state TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_session_sequence
            ON events (session_id, sequence_no);
        CREATE INDEX IF NOT EXISTS idx_quantum_session_step
            ON quantum_states (session_id, evolution_step);
        CREATE INDEX IF NOT EXISTS idx_consensus_session
            ON consensus_snapshots (session_id);
        """

        self.conn.executescript(schema)
        self.conn.commit()

    def create_session(
        self, title: str = "Incident Report War"
    ) -> int:
        session_uuid = str(uuid.uuid4())
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO sessions (session_uuid, title)
                VALUES (?, ?)
                """,
                (session_uuid, title),
            )
        return int(cursor.lastrowid)

    def add_participant(
        self,
        key: str,
        display_name: str,
        ptype: str = "model",
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> int:
        if ptype not in ("human", "model", "system"):
            raise ValueError(f"Invalid participant type: {ptype}")

        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO participants (
                    participant_key,
                    display_name,
                    participant_type,
                    provider,
                    model_name
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, display_name, ptype, provider, model_name),
            )

        participant = self.conn.execute(
            """
            SELECT id FROM participants
            WHERE participant_key = ?
            """,
            (key,),
        ).fetchone()

        if participant is None:
            raise RuntimeError("Failed to create participant")

        return int(participant[0])

    def log_event(
        self,
        session_id: int,
        participant_key: str,
        content: str,
        reply_to: Optional[int] = None,
        event_type: str = "message",
    ) -> int:
        participant = self.conn.execute(
            """
            SELECT id FROM participants
            WHERE participant_key = ?
            """,
            (participant_key,),
        ).fetchone()

        if participant is None:
            raise ValueError(f"Unknown participant: {participant_key}")

        participant_id = int(participant[0])
        content_sha256 = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()
        event_uuid = str(uuid.uuid4())

        self.conn.execute("BEGIN IMMEDIATE")

        try:
            sequence_no = self.conn.execute(
                """
                SELECT COALESCE(MAX(sequence_no), 0) + 1
                FROM events
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()[0]

            cursor = self.conn.execute(
                """
                INSERT INTO events (
                    event_uuid,
                    session_id,
                    participant_id,
                    reply_to_event_id,
                    sequence_no,
                    event_type,
                    content,
                    content_sha256
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_uuid,
                    session_id,
                    participant_id,
                    reply_to,
                    sequence_no,
                    event_type,
                    content,
                    content_sha256,
                ),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        return int(cursor.lastrowid)

    def init_quantum_engine(self, labels: List[str]) -> None:
        self.quantum = QuantumStateEngine(labels)
        print("Quantum-inspired engine initialized:")
        for label in labels:
            print(f"  - {label}")

    def _hamiltonian_hash(self, hamiltonian: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(
            hamiltonian, dtype=np.complex128
        )
        return hashlib.sha256(contiguous.tobytes()).hexdigest()

    def persist_quantum_state(
        self,
        session_id: int,
        evolution_step: int,
        hamiltonian: np.ndarray,
        lambda_val: float,
        dt: float,
    ) -> int:
        if self.quantum is None:
            raise RuntimeError("Quantum engine is not initialized")

        amplitudes = self.quantum.state.amplitudes
        probabilities = self.quantum.observe()
        state_uuid = str(uuid.uuid4())
        hamiltonian_sha256 = self._hamiltonian_hash(hamiltonian)

        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO quantum_states (
                    session_id,
                    state_uuid,
                    labels_json,
                    amplitudes_real_json,
                    amplitudes_imag_json,
                    probabilities_json,
                    most_likely_state,
                    evolution_step,
                    lambda_value,
                    dt_value,
                    hamiltonian_sha256
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    state_uuid,
                    json.dumps(self.quantum.labels),
                    json.dumps(amplitudes.real.tolist()),
                    json.dumps(amplitudes.imag.tolist()),
                    json.dumps(probabilities),
                    self.quantum.most_likely_state(),
                    evolution_step,
                    lambda_val,
                    dt,
                    hamiltonian_sha256,
                ),
            )
        return int(cursor.lastrowid)

    def evolve_log_state(
        self,
        session_id: int,
        hypothesis_potentials: List[float],
        coupling: np.ndarray,
        lambda_val: float = 0.7,
        dt: float = 1.0,
    ) -> Dict[str, float]:
        if self.quantum is None:
            raise RuntimeError("Quantum engine is not initialized")

        hamiltonian = self.quantum.build_hamiltonian(
            np.asarray(hypothesis_potentials, dtype=np.float64),
            coupling,
            lambda_val,
        )

        step = self.conn.execute(
            """
            SELECT COALESCE(MAX(evolution_step), 0) + 1
            FROM quantum_states
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()[0]

        self.quantum.evolve(hamiltonian, dt)
        probabilities = self.quantum.observe()

        self.persist_quantum_state(
            session_id=session_id,
            evolution_step=int(step),
            hamiltonian=hamiltonian,
            lambda_val=lambda_val,
            dt=dt,
        )

        self.persist_consensus_snapshot(
            session_id=session_id,
            engine_type="quantum_inspired",
            probabilities=probabilities,
        )

        return probabilities

    def evaluate_classical(
        self, session_id: int, assessments: np.ndarray
    ) -> Dict[str, float]:
        if self.quantum is None:
            raise RuntimeError("Initialize hypothesis labels first")

        probabilities = self.classical.evaluate(
            self.quantum.labels, assessments
        )

        self.persist_consensus_snapshot(
            session_id=session_id,
            engine_type="classical",
            probabilities=probabilities,
        )
        return probabilities

    def persist_consensus_snapshot(
        self,
        session_id: int,
        engine_type: str,
        probabilities: Dict[str, float],
    ) -> int:
        if engine_type not in ("classical", "quantum_inspired"):
            raise ValueError(f"Invalid engine type: {engine_type}")

        most_likely = max(probabilities, key=probabilities.get)
        snapshot_uuid = str(uuid.uuid4())

        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO consensus_snapshots (
                    session_id,
                    snapshot_uuid,
                    engine_type,
                    probabilities_json,
                    most_likely_state
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    snapshot_uuid,
                    engine_type,
                    json.dumps(probabilities),
                    most_likely,
                ),
            )
        return int(cursor.lastrowid)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MultiAILogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def main() -> None:
    labels = [
        "Purple Correct",
        "Green Correct",
        "Josh Chaos Wins",
        "Mutual Destruction",
    ]

    with MultiAILogger() as logger:
        logger.add_participant("josh", "Josh", ptype="human")
        logger.add_participant(
            "purple", "Sue", ptype="model", provider="OpenAI"
        )
        logger.add_participant(
            "green", "Eve", ptype="model", provider="xAI"
        )
        logger.add_participant(
            "orange", "Claude", ptype="model", provider="Anthropic"
        )

        session_id = logger.create_session(
            "Purple-Green-Orange Consensus Experiment"
        )

        event_1 = logger.log_event(
            session_id, "josh", "A small one."
        )
        event_2 = logger.log_event(
            session_id,
            "purple",
            "A small backstory, or are you making a size joke?",
            reply_to=event_1,
        )
        logger.log_event(
            session_id,
            "josh",
            "Both actually.",
            reply_to=event_2,
        )

        logger.init_quantum_engine(labels)

        assessments = np.array(
            [
                [0.90, 0.20, 0.70, 0.10],
                [0.30, 0.95, 0.80, 0.20],
                [0.50, 0.50, 0.90, 0.30],
            ],
            dtype=np.float64,
        )

        classical = logger.evaluate_classical(
            session_id, assessments
        )

        coupling = disagreement_coupling(assessments)

        evidence_profiles = [
            (0.90, 0.20, 0.10),
            (0.85, 0.25, 0.10),
            (0.95, 0.05, 0.02),
            (0.20, 0.70, 0.20),
        ]

        hypothesis_potentials = [
            evidence_to_potential(support, oppose, uncertainty)
            for support, oppose, uncertainty in evidence_profiles
        ]

        quantum = logger.evolve_log_state(
            session_id=session_id,
            hypothesis_potentials=hypothesis_potentials,
            coupling=coupling,
            lambda_val=0.8,
            dt=1.0,
        )

        print("\nCLASSICAL CONSENSUS")
        for label, probability in classical.items():
            print(f"  {label:<25} {probability:.4f}")

        print("\nQUANTUM-INSPIRED CONSENSUS")
        for label, probability in quantum.items():
            print(f"  {label:<25} {probability:.4f}")

        print("\nCOUPLING MATRIX")
        print(coupling)

        print("\nHYPOTHESIS POTENTIALS")
        for label, potential in zip(labels, hypothesis_potentials):
            print(f"  {label:<25} {potential:.4f}")


if __name__ == "__main__":
    main()
