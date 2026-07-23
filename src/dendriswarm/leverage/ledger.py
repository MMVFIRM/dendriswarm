"""Idempotent integer credit ledger with explicit bond refund and reward cancel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Account:
    available: int = 0
    locked: int = 0
    vested: int = 0


@dataclass(frozen=True)
class LedgerEntry:
    key: str
    contributor: str
    kind: str
    units: int


class LeverageLedger:
    def __init__(self) -> None:
        self.accounts: dict[str, Account] = {}
        self.entries: list[LedgerEntry] = []
        self._keys: set[str] = set()
        self.burned_units = 0
        self.slashed_units = 0
        self.cancelled_reward_units = 0

    def has_key(self, key: str) -> bool:
        return key in self._keys

    def _apply(self, key: str, contributor: str, kind: str, units: int) -> bool:
        if key in self._keys:
            return False
        if units < 0:
            raise ValueError("ledger units must be non-negative")
        self._keys.add(key)
        self.entries.append(LedgerEntry(key, contributor, kind, units))
        return True

    def account(self, contributor: str) -> Account:
        return self.accounts.setdefault(contributor, Account())

    def grant(self, key: str, contributor: str, units: int) -> None:
        if self.has_key(key):
            return
        if self._apply(key, contributor, "grant", units):
            self.account(contributor).available += units

    def burn_fee(self, key: str, contributor: str, units: int) -> None:
        if self.has_key(key):
            return
        account = self.account(contributor)
        if account.available < units:
            raise ValueError("insufficient available balance for fee")
        if self._apply(key, contributor, "fee-burned", units):
            account.available -= units
            self.burned_units += units

    def lock_bond(self, key: str, contributor: str, units: int) -> None:
        if self.has_key(key):
            return
        account = self.account(contributor)
        if account.available < units:
            raise ValueError("insufficient available balance for bond")
        if self._apply(key, contributor, "bond-locked", units):
            account.available -= units
            account.locked += units

    def refund_bond(self, key: str, contributor: str, units: int) -> None:
        if self.has_key(key):
            return
        account = self.account(contributor)
        if account.locked < units:
            raise ValueError("insufficient locked balance to refund")
        if self._apply(key, contributor, "bond-refunded", units):
            account.locked -= units
            account.available += units

    def burn_locked(self, key: str, contributor: str, units: int) -> None:
        if self.has_key(key):
            return
        account = self.account(contributor)
        if account.locked < units:
            raise ValueError("insufficient locked balance to burn")
        if self._apply(key, contributor, "bond-burned", units):
            account.locked -= units
            self.burned_units += units

    def lock_reward(self, key: str, contributor: str, units: int) -> None:
        if self.has_key(key):
            return
        if self._apply(key, contributor, "reward-locked", units):
            self.account(contributor).locked += units

    def cancel_reward(self, key: str, contributor: str, units: int) -> None:
        if self.has_key(key):
            return
        account = self.account(contributor)
        if account.locked < units:
            raise ValueError("insufficient locked reward to cancel")
        if self._apply(key, contributor, "reward-cancelled", units):
            account.locked -= units
            self.cancelled_reward_units += units

    def vest(self, key: str, contributor: str, units: int) -> None:
        if self.has_key(key):
            return
        account = self.account(contributor)
        if account.locked < units:
            raise ValueError("insufficient locked balance to vest")
        if self._apply(key, contributor, "vested", units):
            account.locked -= units
            account.vested += units

    def slash(self, key: str, contributor: str, units: int) -> None:
        if self.has_key(key):
            return
        account = self.account(contributor)
        slashable = min(units, account.locked)
        if self._apply(key, contributor, "slashed", slashable):
            account.locked -= slashable
            self.slashed_units += slashable

    def to_dict(self) -> dict[str, Any]:
        return {
            "accounts": {name: account.__dict__.copy() for name, account in self.accounts.items()},
            "entries": [entry.__dict__.copy() for entry in self.entries],
            "burned_units": self.burned_units,
            "slashed_units": self.slashed_units,
            "cancelled_reward_units": self.cancelled_reward_units,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LeverageLedger":
        ledger = cls()
        ledger.accounts = {name: Account(**account) for name, account in value.get("accounts", {}).items()}
        ledger.entries = [LedgerEntry(**entry) for entry in value.get("entries", [])]
        ledger._keys = {entry.key for entry in ledger.entries}
        ledger.burned_units = int(value.get("burned_units", 0))
        ledger.slashed_units = int(value.get("slashed_units", 0))
        ledger.cancelled_reward_units = int(value.get("cancelled_reward_units", 0))
        return ledger

    def snapshot(self) -> dict[str, Any]:
        return {
            "accounts": {
                contributor: {
                    "available": account.available,
                    "locked": account.locked,
                    "vested": account.vested,
                }
                for contributor, account in sorted(self.accounts.items())
            },
            "burned_units": self.burned_units,
            "slashed_units": self.slashed_units,
            "cancelled_reward_units": self.cancelled_reward_units,
            "entries": len(self.entries),
        }
