import codecs
from typing import Optional, Union
from datetime import datetime, date, time, timedelta
from decimal import Decimal
TWOPLACES = Decimal(10) ** -2
SIXPLACES = Decimal(10) ** -6

from xml.etree import ElementTree as etree
from xml.dom import minidom

from ofxstatement.statement import (
    Statement,
    StatementLine,
    InvestStatementLine,
    BankAccount,
    Currency,
)

import pandas as pd
import sys
import re
import csv

class CsvWriter(object):
    mappings_account = [
        (re.compile(r"X59128643"), "Fidelity:Fidelity X59-128643"),
        (re.compile(r"159258482"), "Fidelity:Fidelity 159258482 (Jason)"),
        (re.compile(r"352042315"), "Fidelity:Fidelity 352042315 (Elisa)"),
    ]

    def __init__(self, statement: Statement) -> None:
        self.statement = statement
        self.genTime = datetime.now()
        self.ld = []
        self.default_float_precision = 2
        self.invest_transactions_float_precision = 5

    def tocsv(self, pretty: bool = False, encoding: str = "utf-8") -> str:
        self.buildDocument()
        return self.csv_statement

    def buildDocument(self) -> list[dict[str, str | float | None]]:
        # Build transaction list as list of dictionaries, using contents of self.statement
        for line in self.statement.invest_lines:
            d = line.__dict__

            self.ld.append(d)

        df_statement = pd.DataFrame(self.ld)

        if self.statement.type == 'fidelity':
            cols = ["Date","Account","Description","TransactionCommodity","Amount","Price","Value", "TransactionID"]
        elif self.statement.type == 'citicard':
            cols = ["Date","Account","Description","Amount","Price","Value","TransactionID"]
        else:
            cols = df_statement.columns

        df_statement = df_statement[cols]

        self.csv_statement = df_statement.to_csv(date_format='%Y-%m-%d', quoting=csv.QUOTE_STRINGS, index=False)
        return

    # Some methods adopted from ofxstatement.ofx
    def buildText(self, tag: str, text: Optional[str], skipEmpty: bool = True) -> None:
        if not text and skipEmpty:
            return
        self.tb.start(tag, {})
        self.tb.data(text or "")
        self.tb.end(tag)

    def buildDateTime(
        self,
        tag: str,
        dt: Optional[datetime],
        skipEmpty: bool = True,
        omitEmptyTime=False,
    ) -> None:
        if not dt and skipEmpty:
            return
        if dt is None:
            self.buildText(tag, "", skipEmpty)
        else:
            utc_offset = dt.utcoffset()

            format = "%Y%m%d"
            if dt.time() != time.min or utc_offset or not omitEmptyTime:
                format += "%H%M%S"
            if dt.microsecond or utc_offset:
                format += f".{(dt.microsecond // 1000):03d}"
            if utc_offset is not None:
                format += f"[{utc_offset.total_seconds() / 3600}]"
            self.buildText(tag, dt.strftime(format))

    def buildAmount(
        self,
        tag: str,
        amount: Optional[Decimal],
        skipEmpty: bool = True,
        precision: Optional[int] = None,
    ) -> None:
        if amount is None and skipEmpty:
            return
        if amount is None:
            self.buildText(tag, "", skipEmpty)
        else:
            if precision is None:
                precision = self.default_float_precision

            self.buildText(tag, "{0:.{precision}f}".format(amount, precision=precision))
