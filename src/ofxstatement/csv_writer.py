import codecs
from typing import Optional, Union
from datetime import datetime, date, time, timedelta
from decimal import Decimal
TWOPLACES = Decimal(10) ** -2

from xml.etree import ElementTree as etree
from xml.dom import minidom

from ofxstatement.statement import (
    Statement,
    StatementLine,
    InvestStatementLine,
    BankAccount,
    Currency,
)

from gncxml_integration import Book, copy_gnucash_accounts
import pandas as pd
import sys
# import uuid
# from typing import Dict, Optional, Any, Iterable, List, TextIO, TypeVar, Generic
# from decimal import Decimal as D
import re
# from abc import abstractmethod
import csv
# from os import path
# from pprint import pformat

class CsvWriter(object):
    mappings_account = [
        (re.compile(r"^X59128643"), "Fidelity:Fidelity X59-128643"),
        (re.compile(r"^159258482"), "Fidelity:Fidelity 159258482 (Jason)"),
        (re.compile(r"^352042315"), "Fidelity:Fidelity 352042315 (Elisa)"),
    ]

    mortgage_pattern = re.compile(r"^DIRECT DEBIT FREEDOM MTG PYMTS")
    dividend_FDRXX_pattern = re.compile(r"^REINVESTMENT FIDELITY GOVERNMENT CASH RESERVES (FDRXX)")
    dividend_FDRXX_pattern = re.compile(r"^REINVESTMENT CASH (315994103)")

    def __init__(self, statement: Statement) -> None:
        self.statement = statement
        self.genTime = datetime.now()
        self.ld = []
        self.default_float_precision = 2
        self.invest_transactions_float_precision = 5

        self.book = self.initialize_book()
        self.mortgage_account = "Real Estate:Mortgage Amerisave"
        self.interest_account = "Interest:Mortgage"
        self.escrow_account = "Real Estate:Escrow Amerisave"
        self.mortgage_rate = Decimal(2.75)
        self.mortgage_principal_interest = Decimal(2570.94)

        # print(f"self.book.accounts.columns = {self.book.accounts.columns}")
        # print(f"{self.book.accounts}")
        # print(f"self.book.splits.columns = {self.book.splits.columns}")
        # print(f"{self.book.splits}")
        # print(f"self.book.transactions.columns = {self.book.transactions.columns}")
        # print(f"{self.book.transactions}")

    def tocsv(self, pretty: bool = False, encoding: str = "utf-8") -> str:
        self.buildDocument()
        return self.csv_statement

    def buildDocument(self) -> list[dict[str, str | float | None]]:
        # Build transaction list as list of dictionaries, using contents of self.statement
        for line in self.statement.invest_lines:
            d = line.__dict__

            for pattern, name in self.mappings_account:
                if pattern.match(d["account"]):
                    d["account"] = name
                    break

            self.ld.append(d)
            # print(f"buildDocument:  d = {d}")

            mortgage_match = self.mortgage_pattern.match(d["memo"])
            if mortgage_match:
                self.buildMortgage(d)

        df_statement = pd.DataFrame(self.ld)
        df_statement = df_statement.set_index("id")

        df_cols = df_statement.columns
        cols = ["date", "account_type","account","memo","trntype","trntype_detailed","security_id","units","unit_price","amount"]
        for col in cols:
            if col not in df_cols:
                df_statement[col] = pd.Series()
        df_cols = df_statement.columns
        newcols = [col for col in cols if col in df_cols] + [col for col in df_cols if col not in cols]
        df_statement = df_statement[newcols]
        
        # self.csv_statement = df_statement.to_csv(index=False, date_format='%Y-%m-%d')
        self.csv_statement = df_statement.to_csv(date_format='%Y-%m-%d', quoting=csv.QUOTE_STRINGS)
        return

    def buildMortgage(self, d):        
        mortgage_payment = -Decimal(d["amount"]).quantize(TWOPLACES)
        mortgage_balance = (-self.account_balance(self.book,  self.mortgage_account, d["date"] + timedelta(days=-1))).quantize(TWOPLACES)
        mortgage_interest = Decimal(mortgage_balance * self.mortgage_rate / 1200).quantize(TWOPLACES)
        mortgage_principal = (self.mortgage_principal_interest - mortgage_interest).quantize(TWOPLACES)
        mortgage_escrow = (mortgage_payment - self.mortgage_principal_interest).quantize(TWOPLACES)
        mortgage_delta = mortgage_payment - mortgage_principal - mortgage_interest - mortgage_escrow
        if mortgage_delta != Decimal(0):
            print(f"buildMortgage:  Error, mortgage payment not sum of principal, interest and escrow")
        # print(f"buildMortgage:  mortgage_balance = {mortgage_balance}, mortgage_payment = {mortgage_payment}, mortgage_principal = {mortgage_principal}, mortgage_interest = {mortgage_interest}, mortgage_escrow = {mortgage_escrow}, payment = {mortgage_principal + mortgage_interest + mortgage_escrow}\n")

        d_principal = d.copy()
        d_principal["account"] = self.mortgage_account
        d_principal["amount"] = mortgage_principal
        self.ld.append(d_principal)
        # print(f"\nbuildMortgage:  d = {d}\n")
        # print(f"buildMortgage:  d_principal = {d_principal}\n")

        d_interest = d.copy()
        d_interest["account"] = self.interest_account
        d_interest["amount"] = mortgage_interest
        self.ld.append(d_interest)
        # print(f"buildMortgage:  d = {d}")
        # print(f"buildMortgage:  d_interest = {d_interest}\n")

        d_escrow = d.copy()
        d_escrow["account"] = self.escrow_account
        d_escrow["amount"] = mortgage_escrow

        self.ld.append(d_escrow)
        # print(f"buildDocument:  d = {d}")
        # print(f"buildMortgage:  d_escrow = {d_escrow}\n")
        
    def initialize_book(self) -> Book:
        bookname = copy_gnucash_accounts()

        try:
            book = Book(bookname)
        except OSError as err:
            sys.exit(err)

        return book

    def account_balance(self, book, account, date = None):
        # print(f"account_balance:  account = {account}, date = {date}")
        id = book.accounts[book.accounts['path'] == account].index.values[0][1]
        # print(f"account_balance:  id = {id}")
        splits = book.splits[book.splits['act_id'] == id].sort_values(by='trn_date')
        splits['balance'] = splits['value'].cumsum()
    
        if date is None:
            balance = splits['balance'].iloc[-1]
        else:
            balance = splits[splits['trn_date'] <= date]['balance'].iloc[-1]

        return balance

    def make_multisplit_transaction(
        self, 
        transaction_id: str,
        account: str,
        date: str,
        description: str,
        amount: float,
        value: float,
        commodity: str | None,
        price: float | None = None,
        balance: float | None = None,
    ) -> dict[str, str | float | None]:

        price = price if price is not None else 1

        return {
            "TransactionId": transaction_id,
            "Account": account,
            "Date": date,
            "Description": description,
            "Amount": amount,
            "Value": value,
            "Commodity/Currency": commodity,
            "Price/Rate": 1 if price is None else price,
            "Balance": balance,
        }

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
