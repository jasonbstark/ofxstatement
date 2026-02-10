from decimal import Decimal
from fractions import Fraction
import collections
import gzip
import re
import xml.etree.ElementTree as ET
import pandas as pd
import gncxml._iso4217 as iso4217
import copy
import numpy as np
import collections
import datetime as dt
from scipy import optimize
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import math
import shutil
from IPython.display import Markdown as md
# import ipdb

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=RuntimeWarning)
simplefilter(action="ignore", category=FutureWarning)

import logging
logger = logging.getLogger('yfinance')
logger.disabled = True
logger.propagate = False

def copy_gnucash_accounts():
    src = "/Volumes/Nextcloud Data/Nextcloud/Jason & Elisa's documents/finance/accounts/Stark Caridi accounts.gnucash"
    date_string = dt.datetime.now().strftime('%Y %m %d %H.%M.%S')
    file_string = f"gnucash/Stark Caridi accounts XML {date_string}.gnucash"
    dst = f"/Volumes/Nextcloud Data/Nextcloud/Jason & Elisa's documents/finance/financial modeling/investment returns/repo/investment returns/{file_string}"        

    shutil.copy2(src, dst)
    
    return file_string

# XML file parsing methods adapted from https://github.com/LiosK/gncxml.git
class Book:
    """Parse GnuCash XML data file and provide interface to read journal entries and master data tables."""
    
    def __init__(self, gncfile):
        """
        Parameters
        ----------
        gncfile :   file name or file object (io.BufferedReader) GnuCash data file (XML format)
        members :   DataFrames are constructed for accounts, commodities, prices, transactions and splits;
                    an empty portfolio dictionary is established, which is used by objects of the Portfolio class,
                    to store references to Portfolio objects.  
        """
        if hasattr(gncfile, "buffer"):
            # in case io.TextIOWrapper is passed
            gncfile = gncfile.buffer

        gzmagic = b"\x1f\x8b"
        if hasattr(gncfile, "peek"):
            gzipped = gncfile.peek(2).startswith(gzmagic)
        else:
            with open(gncfile, "rb") as source:
                gzipped = source.peek(2).startswith(gzmagic)

        if gzipped:
            with gzip.open(gncfile) as source:
                self._tree = ET.parse(source)
        else:
            self._tree = ET.parse(gncfile)

        self._ns = {
            "act": "http://www.gnucash.org/XML/act",
            "cmdty": "http://www.gnucash.org/XML/cmdty",
            "gnc": "http://www.gnucash.org/XML/gnc",
            "price": "http://www.gnucash.org/XML/price",
            "split": "http://www.gnucash.org/XML/split",
            "trn": "http://www.gnucash.org/XML/trn",
            "ts": "http://www.gnucash.org/XML/ts",
        }

        self.bookname = gncfile
        self.commodities = self.__init_commodities()
        self.accounts = self.__init_accounts()
        self.prices = self.__init_prices()
        self.transactions = self.__init_transactions()
        self.splits = self.__init_splits()
        self.__init_accounts_bank()
        self.portfolios = {}

        bookdate = dt.datetime.now()
        self.bookdate = bookdate

    def __init_commodities(self):
        """Return commodity (aka currency or equity) entries as pandas.DataFrame."""
        tx = self._findtext_wrapper(self._ns)

        items = []
        dec = re.compile(r"^10*$")
        for e in self._tree.findall("./gnc:book/gnc:commodity", self._ns):
            space = tx(e, "./cmdty:space", "")
            cmdid = tx(e, "./cmdty:id")
            crncy = {"name": None, "fraction": None}
            if space in {"CURRENCY", "ISO4217"}:
                crncy = iso4217.get(cmdid, crncy)
            frac = tx(e, "./cmdty:fraction", crncy["fraction"])
            exponent = None
            if frac is not None and dec.match(frac):
                exponent = len(frac) - 1.0  # float to treat NaN
            items.append(
                {
                    "space": space,
                    "id": cmdid,
                    "name": tx(e, "./cmdty:name", crncy["name"]),
                    "xcode": tx(e, "./cmdty:xcode"),
                    "fraction": frac,
                    "exponent": exponent,
                    "quote_source": tx(e, "./cmdty:quote_source"),
                }
            )

        return pd.DataFrame(
            items,
            columns=[
                "space",
                "id",
                "name",
                "xcode",
                "fraction",
                "exponent",
                "quote_source",
            ],
        ).set_index(["space", "id"])

    def __init_accounts(self):
        """Return account entries as pandas.DataFrame."""
        tx = self._findtext_wrapper(self._ns)

        items = collections.OrderedDict()
        for e in self._tree.findall("./gnc:book/gnc:account", self._ns):
            cur = e.find("./act:id", self._ns)
            cur_key = (cur_idtype, cur_id) = (cur.attrib.get("type", ""), cur.text)
            parent = e.find("./act:parent", self._ns)
            if parent is not None:
                parent = (parent.attrib.get("type", ""), parent.text)
            items[cur_key] = {
                "idtype": cur_idtype,
                "id": cur_id,
                "name": tx(e, "./act:name"),
                "type": tx(e, "./act:type"),
                "code": tx(e, "./act:code"),
                "description": tx(e, "./act:description"),
                "cmd_space": tx(e, "./act:commodity/cmdty:space", ""),
                "cmd_id": tx(e, "./act:commodity/cmdty:id"),
                "parent": parent,
            }

        def retrieve_path(e):
            """
            Parameters
            ----------
            e : construct 'path', 'toplevel' and 'parent_path' expressions, to identify account structure
            """
            if "toplevel" in e:  # already set
                return e
            if e["parent"] is None:  # root
                e["path"] = e["toplevel"] = e["parent_path"] = None
                return e
            parent = retrieve_path(items[e["parent"]])
            if parent["toplevel"] is None:  # toplevel
                e["path"] = e["toplevel"] = e["name"]
                e["parent_path"] = None
            else:
                e["path"] = parent["path"] + ":" + e["name"]
                e["toplevel"] = parent["toplevel"]
                e["parent_path"] = parent["path"]
            return e
            
        df = pd.DataFrame(
            [retrieve_path(e) for e in items.values()],
            columns=[
                "idtype",
                "id",
                "path",
                "toplevel",
                "parent_path",
                "name",
                "type",
                "code",
                "description",
                "cmd_space",
                "cmd_id",
            ],
        ).set_index(["idtype", "id"])
        
        cmds = self.commodities.add_prefix("cmd_")
        df = df.join(cmds, ["cmd_space", "cmd_id"])
        
        return df

    def __init_prices(self):
        """Return commodity price entries as pandas.DataFrame."""
        tx = self._findtext_wrapper(self._ns)

        items = []
        for e in self._tree.findall("./gnc:book/gnc:pricedb/price", self._ns):
            val = Fraction(tx(e, "./price:value"))
            items.append(
                {
                    "time": pd.Timestamp(tx(e, "./price:time/ts:date")),
                    "cmd_space": tx(e, "./price:commodity/cmdty:space", ""),
                    "cmd_id": tx(e, "./price:commodity/cmdty:id"),
                    "crncy_space": tx(e, "./price:currency/cmdty:space", ""),
                    "crncy_id": tx(e, "./price:currency/cmdty:id"),
                    "source": tx(e, "./price:source"),
                    "type": tx(e, "./price:type", "unknown"),
                    "value": Decimal(val.numerator) / Decimal(val.denominator),
                    "value_frac": val,
                }
            )

        df = pd.DataFrame(
            items,
            columns=[
                "time",
                "cmd_space",
                "cmd_id",
                "crncy_space",
                "crncy_id",
                "source",
                "type",
                "value",
                "value_frac",
            ],
        )
        
        cmds = self.commodities
        df = df.join(cmds.add_prefix("cmd_"), ["cmd_space", "cmd_id"]).join(cmds.add_prefix("crncy_"), ["crncy_space", "crncy_id"])
        return df

    def __init_transactions(self):
        """Return transaction (aka header) entries as pandas.DataFrame."""
        tx = self._findtext_wrapper(self._ns)

        items = []
        for e in self._tree.findall("./gnc:book/gnc:transaction", self._ns):
            trnid = e.find("./trn:id", self._ns)
            items.append(
                {
                    "idtype": trnid.attrib.get("type", ""),
                    "id": trnid.text,
                    "date": pd.Timestamp(
                        tx(e, "./trn:date-posted/ts:date").split(" ")[0]
                    ),
                    "num": tx(e, "./trn:num"),
                    "description": tx(e, "./trn:description"),
                    "crncy_space": tx(e, "./trn:currency/cmdty:space", ""),
                    "crncy_id": tx(e, "./trn:currency/cmdty:id"),
                }
            )

        df = pd.DataFrame(
            items,
            columns=[
                "idtype",
                "id",
                "date",
                "num",
                "description",
                "crncy_space",
                "crncy_id",
            ],
        ).set_index(["idtype", "id"])
        
        cmds = self.commodities.add_prefix("crncy_")

        return df.join(cmds, ["crncy_space", "crncy_id"])

    def __init_splits(self):
        """Return split (aka line item) entries as flat pandas.DataFrame after joining relevant tables."""
        tx = self._findtext_wrapper(self._ns)

        items = []
        for e in self._tree.findall("./gnc:book/gnc:transaction", self._ns):
            trnid = e.find("./trn:id", self._ns)
            header = {
                "trn_idtype": trnid.attrib.get("type", ""),
                "trn_id": trnid.text,
            }
            for sp in e.findall("./trn:splits/trn:split", self._ns):
                spid = sp.find("./split:id", self._ns)
                actid = sp.find("./split:account", self._ns)
                val = Fraction(tx(sp, "./split:value"))
                qty = Fraction(tx(sp, "./split:quantity"))
                items.append(
                    {
                        "idtype": spid.attrib.get("type", ""),
                        "id": spid.text,
                        "action": tx(sp, "./split:action"),
                        "memo": tx(sp, "./split:memo"),
                        "reconciled": tx(sp, "./split:reconciled-state"),
                        "value": Decimal(val.numerator) / Decimal(val.denominator),
                        "value_frac": val,
                        "quantity": Decimal(qty.numerator) / Decimal(qty.denominator),
                        "quantity_frac": qty,
                        "act_idtype": actid.attrib.get("type", ""),
                        "act_id": actid.text,
                        **header,
                    }
                )

        df = pd.DataFrame(
            items,
            columns=[
                "idtype",
                "id",
                "action",
                "memo",
                "reconciled",
                "value",
                "value_frac",
                "quantity",
                "quantity_frac",
                "act_idtype",
                "act_id",
                "trn_idtype",
                "trn_id",
            ],
        ).set_index(["idtype", "id"])

        acts = self.accounts.add_prefix("act_")
        trns = self.transactions.add_prefix("trn_")
        return (
            df
            .join(acts, ["act_idtype", "act_id"])
            .join(trns, ["trn_idtype", "trn_id"])
        )

    def __init_accounts_bank(self):
        self.accounts_bank = {}
        account_numbers = ['1014111292255', '3931507721', '8738606899']
        for account_number in account_numbers:
            account_path = 'Wells Fargo:Wells Fargo ' + account_number
            account = self.accounts[self.accounts['path'] == account_path]
            account_id = account.index[0][1]
            account_id_splits = self.splits[self.splits['act_id'] == account_id].sort_values(by=['trn_date'])
            account_cashflows = account_id_splits.copy().loc[:, account_id_splits.columns]
            account_cashflows.loc[:, 'balance'] = account_cashflows.loc[:, 'value'].cumsum()

            account_cashflows = account_cashflows[['trn_date', 'value', \
                   'balance', 'action', 'memo', 'reconciled', 'value_frac', 'quantity', \
                   'quantity_frac', 'act_idtype', 'act_id', 'trn_idtype', 'trn_id', \
                   'act_path', 'act_toplevel', 'act_parent_path', 'act_name', 'act_type', \
                   'act_code', 'act_description', 'act_cmd_space', 'act_cmd_id', \
                   'act_cmd_name', 'act_cmd_xcode', 'act_cmd_fraction', 'act_cmd_exponent', \
                   'act_cmd_quote_source', 'trn_num', 'trn_description', \
                   'trn_crncy_space', 'trn_crncy_id', 'trn_crncy_name', 'trn_crncy_xcode', \
                   'trn_crncy_fraction', 'trn_crncy_exponent', 'trn_crncy_quote_source']]

            self.accounts_bank[account_number] = account_cashflows

        self.accounts_bank_daily = self.build_accounts_bank_daily()
        [self.accounts_bank_consolidated, self.accounts_bank_summary] = self.build_accounts_bank_consolidated()

    def bank_balance(self, date_balance=pd.Timestamp(2020, 2, 27)):
        account_numbers = ['Wells Fargo:Wells Fargo 1014111292255', 'Wells Fargo:Wells Fargo 3931507721', 'Wells Fargo:Wells Fargo 8738606899']
        balances = []
        balance_total = 0.
        for account_number in account_numbers:
            account = self.accounts[self.accounts['path'] == account_number]
            account_id = account.index[0][1]
            account_id_splits = self.splits[self.splits['act_id'] == account_id].sort_values(by=['trn_date'])
            account_cashflows = account_id_splits.copy().loc[:, account_id_splits.columns]
            account_cashflows.loc[:, 'balance'] = account_cashflows.loc[:, 'value'].cumsum()
            account_cashflows_balance = float(account_cashflows[(account_cashflows['trn_date'] <= date_balance)][['trn_date', 'value', 'balance']].iloc[-1]['balance'])
            balance_total += account_cashflows_balance
            balances.append(account_cashflows_balance)
        balances.insert(0, balance_total)
        return balances

    def build_accounts_bank_daily(self):
        account_numbers = ['1014111292255', '3931507721', '8738606899']
        date_start_accounts = self.accounts_bank[account_numbers[0]]['trn_date'].iloc[0]
        date_end_accounts = pd.to_datetime('today').normalize()
    
        for account_number in account_numbers[1:]:
            date_start_accounts = min(date_start_accounts, self.accounts_bank[account_number]['trn_date'].iloc[0])
    
        dates_consecutive = pd.to_datetime(pd.date_range(start=date_start_accounts, end=date_end_accounts))
    
        accounts_bank = {}
        for account_number in account_numbers:
            columns = ['Date', 'act_name', 'trn_description', 'Value '+account_number, 'Balance '+account_number, 'act_id', 'trn_id', 'value_frac', 'quantity', 'quantity_frac']
            account = self.accounts_bank[account_number].copy().reset_index().rename(columns={"trn_date": "Date", "value": "Value "+account_number,"balance": "Balance "+account_number})[columns]
            date_account_first = account['Date'].iloc[0].date()
    
            transactions = []
            account_index = 0
            date_index = 0
            for datetime in dates_consecutive:
                date = datetime.date()
                if account_index == len(account):
                    data = {'Date': date, 'act_name': row['act_name'], 'trn_description': 'Valuation', 'Value '+account_number: 0., \
                        'Balance '+account_number: float(row['Balance '+account_number]), 'act_id': row['act_id'], 'trn_id': np.nan, 'value_frac': np.nan, \
                        'quantity': np.nan, 'quantity_frac': np.nan}
                    series = pd.Series(data=data, index=None, name=date_index)
    
                    transactions.append(series)
                    date_index += 1
                elif date < date_account_first:
                    row = account.iloc[0]
                    data = {'Date': date, 'act_name': row['act_name'], 'trn_description': 'Valuation', 'Value '+account_number: 0., \
                        'Balance '+account_number: 0., 'act_id': row['act_id'], 'trn_id': np.nan, 'value_frac': np.nan, \
                        'quantity': np.nan, 'quantity_frac': np.nan}
                    series = pd.Series(data=data, index=None, name=date_index)
    
                    transactions.append(series)
                    date_index += 1
                else:
                    row = account.iloc[account_index]
                    row = row.rename(date_index)
                    date_row = row['Date'].date()
    
                    if date_row == date:
                        while date_row == date:
                            row.loc['Date'] = date_row
                            row = row.rename(date_index)
                            row['Value '+account_number] = float(row['Value '+account_number])
                            row['Balance '+account_number] = float(row['Balance '+account_number])
                        
                            transactions.append(row)
                            account_index += 1
                            date_index += 1
                        
                            if account_index < len(account):
                                row_prior = row
                                row = account.iloc[account_index]
                                row = row.rename(date_index)
                                row['Value '+account_number] = float(row['Value '+account_number])
                                row['Balance '+account_number] = float(row['Balance '+account_number])
                                date_row = row['Date'].date()
                            else:
                                break
                    else:
                        data = {'Date': date, 'act_name': row_prior['act_name'], 'trn_description': 'Valuation', 'Value '+account_number: 0., \
                            'Balance '+account_number: float(row_prior['Balance '+account_number]), 'act_id': row['act_id'], 'trn_id': np.nan, 'value_frac': np.nan, \
                            'quantity': np.nan, 'quantity_frac': np.nan}
                        series = pd.Series(data=data, index=None, name=date_index)
    
                        transactions.append(series)
                        date_index += 1
    
            df_daily_indexed = pd.DataFrame(data=transactions)
            accounts_bank[account_number] = df_daily_indexed

        return accounts_bank

    def build_accounts_bank_consolidated(self):
        accounts_bank_consolidated = pd.concat([self.accounts_bank_daily['1014111292255'], self.accounts_bank_daily['3931507721'], self.accounts_bank_daily['8738606899']], ignore_index=True)
        accounts_bank_consolidated = accounts_bank_consolidated.sort_values(by='Date', kind='stable').reset_index(drop=True)
        accounts_bank_consolidated = accounts_bank_consolidated.ffill().fillna(value=0.)
        accounts_bank_consolidated['Balance consolidated'] = accounts_bank_consolidated['Balance 1014111292255'] + accounts_bank_consolidated['Balance 3931507721'] + accounts_bank_consolidated['Balance 8738606899'] 
        accounts_bank_consolidated = accounts_bank_consolidated[['Date', 'act_name', 'trn_description', 'Value 1014111292255', 'Balance 1014111292255', 'Value 3931507721', \
                                                                 'Balance 3931507721', 'Value 8738606899', 'Balance 8738606899', 'Balance consolidated', \
                                                                 'act_id', 'trn_id', 'value_frac', 'quantity', 'quantity_frac']]
        mask = accounts_bank_consolidated['trn_description'] == 'Valuation'
        accounts_bank_consolidated.loc[mask, ['act_name', 'act_id', 'trn_id', 'value_frac', 'quantity', 'quantity_frac']] = np.nan

        mask = (accounts_bank_consolidated['Date'] == accounts_bank_consolidated.shift(periods=-1)['Date']) & (accounts_bank_consolidated['trn_description'] == 'Valuation')
        accounts_bank_consolidated = accounts_bank_consolidated.drop(accounts_bank_consolidated[mask].index)

        mask = (accounts_bank_consolidated['Date'] == accounts_bank_consolidated.shift(periods=1)['Date']) & (accounts_bank_consolidated['trn_description'] == 'Valuation')
        accounts_bank_consolidated = accounts_bank_consolidated.drop(accounts_bank_consolidated[mask].index)

        mask = (accounts_bank_consolidated['Date'] == accounts_bank_consolidated.shift(periods=-1)['Date'])
        accounts_bank_summary = accounts_bank_consolidated.copy().drop(accounts_bank_consolidated[mask].index).reset_index(drop=True)
        accounts_bank_summary = accounts_bank_summary.drop(columns=['act_name', 'trn_description', 'act_id', 'trn_id', 'value_frac', 'quantity', 'quantity_frac'])

        accounts_bank_summary = accounts_bank_summary.rename(columns={'Value 1014111292255': 'Cashflow 1014111292255', 'Value 3931507721': 'Cashflow 3931507721', 'Value 8738606899': 'Cashflow 8738606899'})        
        accounts_bank_summary['Cashflow 1014111292255'] = accounts_bank_summary['Balance 1014111292255'] - accounts_bank_summary.shift(periods=1)['Balance 1014111292255']
        accounts_bank_summary['Cashflow 3931507721'] = accounts_bank_summary['Balance 3931507721'] - accounts_bank_summary.shift(periods=1)['Balance 3931507721']
        accounts_bank_summary['Cashflow 8738606899'] = accounts_bank_summary['Balance 8738606899'] - accounts_bank_summary.shift(periods=1)['Balance 8738606899']
        accounts_bank_summary.loc[0, 'Cashflow 1014111292255'] = accounts_bank_summary.loc[0, 'Balance 1014111292255']
        accounts_bank_summary.loc[0, 'Cashflow 3931507721'] = accounts_bank_summary.loc[0, 'Balance 3931507721']
        accounts_bank_summary.loc[0, 'Cashflow 8738606899'] = accounts_bank_summary.loc[0, 'Balance 8738606899']

        accounts_bank_summary['Cashflow consolidated'] = accounts_bank_summary['Cashflow 1014111292255'] + accounts_bank_summary['Cashflow 3931507721'] + accounts_bank_summary['Cashflow 8738606899'] 
        accounts_bank_summary = accounts_bank_summary[['Date', 'Cashflow 1014111292255', 'Cashflow 3931507721', 'Cashflow 8738606899', 'Balance 1014111292255', 'Balance 3931507721', 'Balance 8738606899', 'Cashflow consolidated', 'Balance consolidated']]
        
        return [accounts_bank_consolidated, accounts_bank_summary]

    def _findtext_wrapper(self, ns):
        return lambda elem, path, default=None: elem.findtext(path, default, ns)

    def mask_cashflows(self, values, date_beginning, date_ending):
        mask = (values['Date'] == date_beginning) \
                | ( \
                    (values['Date'] > date_beginning) \
                    & ( \
                        (values['Date'] < date_ending) \
                        & (values['Cashflow Bank'] != 0.) \
                    ) 
                    | ( \
                        (values['Date'] >= date_ending) \
                        & (values['Cashflow Portfolio Jason'] != 0.) \
                    ) \
                    | ( \
                        (values['Exclusive Jason'] != 0.) \
                    ) \
                    | ( \
                        (values['Exclusive Elisa'] != 0.) \
                    ) \
                )
        return mask

    def fraction_critical(self, portfolio, inheritance_Jason_before_tax, inheritance_Elisa_before_tax):
        cashflows_baseline = portfolio.cashflows_portfolio.copy()
        outflows_total = -cashflows_baseline[cashflows_baseline['Cashflow Portfolio'] < 0].reset_index(drop=True)['Cashflow Portfolio'].sum()
        fraction_Elisa = inheritance_Elisa_before_tax / (outflows_total + inheritance_Jason_before_tax + inheritance_Elisa_before_tax)
        fraction_Jason = 1 - fraction_Elisa
        return fraction_Elisa, fraction_Jason

    def track_inheritances(self, portfolio_Jason, inheritance_Jason_before_tax, portfolio_Elisa, inheritance_Elisa_before_tax, fraction_Jason=None, tax_rate=0.2, exclusions=None, \
                                col_date='Date', col_cashflows='Cashflow Portfolio', col_values='Value', col_gains_date='Date', col_gains='Forward Gain Cumulative', \
                                date_final=None, debug=0):
        use_static_fractions = fraction_Jason is not None

        accounts_bank_summary = self.accounts_bank_summary.copy()

        cashflows_Jason_baseline = portfolio_Jason.cashflows_portfolio.copy()
        cashflows_Jason_baseline = cashflows_Jason_baseline[cashflows_Jason_baseline[col_cashflows] != 0]
        cashflows_Jason_baseline = cashflows_Jason_baseline.groupby([col_date], as_index=False).aggregate({col_cashflows: 'sum'})
        cashflows_Jason_in = cashflows_Jason_baseline[cashflows_Jason_baseline[col_cashflows] > 0].reset_index(drop=True)
    
        cashflows_out = cashflows_Jason_baseline[cashflows_Jason_baseline[col_cashflows] < 0].reset_index(drop=True)

        date_first_outflow = cashflows_out.iloc[0][col_date]

        cashflows_Elisa_baseline = portfolio_Elisa.cashflows_portfolio.copy()
        cashflows_Elisa_baseline = cashflows_Elisa_baseline[cashflows_Elisa_baseline[col_cashflows] != 0]
        cashflows_Elisa_baseline = cashflows_Elisa_baseline.groupby([col_date], as_index=False).aggregate({col_cashflows: 'sum'})
        cashflows_Elisa_in = cashflows_Elisa_baseline[cashflows_Elisa_baseline[col_cashflows] > 0].reset_index(drop=True)
    
        date_start = portfolio_Jason.valuations[col_date].iloc[0]

        values_Jason = portfolio_Jason.valuations[[col_gains_date, col_gains]].copy()
        values_Jason[col_cashflows] = 0.
        values_Jason[col_values] = 0.
        values_Jason = values_Jason[[col_gains_date, col_cashflows, col_gains, col_values]]

        date_before_first_outflow = values_Jason[values_Jason[col_date] < date_first_outflow].iloc[-1][col_date]

        values_Elisa = portfolio_Elisa.valuations[[col_gains_date, col_gains]].copy()
        values_Elisa[col_cashflows] = 0.
        values_Elisa[col_values] = 0.
        values_Elisa = values_Elisa[[col_gains_date, col_cashflows, col_gains, col_values]]

        values = values_Jason[values_Jason[col_date] >= date_start].copy()
        values = values.rename(columns={col_cashflows: 'Cashflow Portfolio Jason', col_gains: 'Forward Gain Cumulative Jason', col_values: 'Value Jason'})
        values[['Cashflow Portfolio Elisa', 'Forward Gain Cumulative Elisa', 'Value Elisa']] = np.nan

        mask = pd.to_datetime(self.accounts_bank_summary[col_date]) >= pd.to_datetime(values[col_date]).iloc[0]
        values['Balance Bank'] = self.accounts_bank_summary[mask]['Balance consolidated'].values
        values['Cashflow Bank'] = self.accounts_bank_summary[mask]['Cashflow consolidated'].values
        values = values[[col_date, 'Cashflow Bank', 'Balance Bank', 'Cashflow Portfolio Jason', 'Forward Gain Cumulative Jason', 'Value Jason', 'Cashflow Portfolio Elisa', 'Forward Gain Cumulative Elisa', 'Value Elisa']]
        values.loc[0, 'Cashflow Bank'] = values.loc[0]['Balance Bank']

        for index, row in cashflows_Jason_in.iterrows():
            date = cashflows_Jason_in.loc[index, col_date]
            cashflow = cashflows_Jason_in.loc[index, col_cashflows]

            date_gains_Jason = values_Jason[values_Jason[col_gains_date] >= date][col_gains_date].min()
            mask_Jason = values_Jason[col_gains_date] == date_gains_Jason
            values_Jason.loc[mask_Jason, col_cashflows] = cashflow
            gain_present_Jason = values_Jason.loc[mask_Jason, col_gains].values[0]

            mask_Jason = values_Jason[col_gains_date] >= date_gains_Jason
            cashflows_future_Jason = cashflow * values_Jason[mask_Jason][col_gains]
            values_Jason.loc[mask_Jason, col_values] += cashflows_future_Jason / gain_present_Jason

        for index, row in cashflows_Elisa_in.iterrows():
            date = cashflows_Elisa_in.loc[index, col_date]
            cashflow = cashflows_Elisa_in.loc[index, col_cashflows]

            date_gains_Elisa = values_Elisa[values_Elisa[col_gains_date] >= date][col_gains_date].min()
            mask_Elisa = values_Elisa[col_gains_date] == date_gains_Elisa
            values_Elisa.loc[mask_Elisa, col_cashflows] = cashflow
            gain_present_Elisa = values_Elisa.loc[mask_Elisa, col_gains].values[0]

            mask_Elisa = values_Elisa[col_gains_date] >= date_gains_Elisa
            cashflows_future_Elisa = cashflow * values_Elisa[mask_Elisa][col_gains]
            values_Elisa.loc[mask_Elisa, col_values] += cashflows_future_Elisa / gain_present_Elisa

        for index, row in cashflows_out.iterrows():
            date = cashflows_out.loc[index, col_date]
            cashflow = cashflows_out.loc[index, col_cashflows]

            date_gains_Jason = values_Jason[values_Jason[col_gains_date] >= date][col_gains_date].min()
            mask_Jason = values_Jason[col_gains_date] == date_gains_Jason
            values_Jason.loc[mask_Jason, col_cashflows] = cashflow
            gain_present_Jason = values_Jason.loc[mask_Jason, col_gains].values[0]

            mask_Jason = values_Jason[col_gains_date] >= date_gains_Jason
            cashflows_future_Jason = cashflow * values_Jason[mask_Jason][col_gains]
            values_Jason.loc[mask_Jason, col_values] += cashflows_future_Jason / gain_present_Jason

        values['Cashflow Portfolio Jason'] = values_Jason[col_cashflows]
        values['Forward Gain Cumulative Jason'] = values_Jason[col_gains]
        values['Value Jason'] = values_Jason[col_values]

        mask = values[col_date] < values_Elisa[col_date].iloc[0]
        values.loc[mask, 'Cashflow Portfolio Elisa'] = 0.
        values.loc[mask, 'Forward Gain Cumulative Elisa'] = 1.
        values.loc[mask, 'Value Elisa'] = 0.

        mask = values[col_date] >= values_Elisa[col_date].iloc[0]
        values.loc[mask, 'Cashflow Portfolio Elisa'] = values_Elisa[col_cashflows].values
        values.loc[mask, 'Forward Gain Cumulative Elisa'] = values_Elisa[col_gains].values
        values.loc[mask, 'Value Elisa'] = values_Elisa[col_values].values
    
        values['Inheritance Jason in IRA Jason'] = 0.
        values['Cumulative Inheritance Jason in IRA Jason'] = 0.
        values['Inheritance Jason in IRA Elisa'] = 0.
        values['Cumulative Inheritance Jason in IRA Elisa'] = 0.

        values['Inheritance Elisa in IRA Jason'] = 0.
        values['Cumulative Inheritance Elisa in IRA Jason'] = 0.
        values['Inheritance Elisa in IRA Elisa'] = 0.
        values['Cumulative Inheritance Elisa in IRA Elisa'] = 0.

        values['Cashflow Jason excess'] = 0.
        values['Cumulative Cashflow Jason excess'] = 0.

        values['Share Jason'] = 0.
        values['Cashflow Jason'] = 0.
        values['Cumulative Cashflow Jason'] = 0.
        values['Exclusive Jason'] = 0.
        values['Cumulative Exclusive Jason'] = 0.

        values['Share Elisa'] = 0.
        values['Cashflow Elisa'] = 0.
        values['Cumulative Cashflow Elisa'] = 0.
        values['Exclusive Elisa'] = 0.
        values['Cumulative Exclusive Elisa'] = 0.

        values['Delta'] = 0.
        values['Cumulative Delta'] = 0.

        values = values[[col_date, 'Cashflow Bank', 'Balance Bank', \
                                    'Share Jason', 'Cashflow Jason', 'Cumulative Cashflow Jason', 'Exclusive Jason', 'Cumulative Exclusive Jason', \
                                    'Share Elisa', 'Cashflow Elisa', 'Cumulative Cashflow Elisa', 'Exclusive Elisa', 'Cumulative Exclusive Elisa', \
                                    'Delta', 'Cumulative Delta', \
                                    'Inheritance Jason in IRA Jason', 'Inheritance Jason in IRA Elisa', 'Inheritance Elisa in IRA Jason', \
                                    'Inheritance Elisa in IRA Elisa', 'Cumulative Inheritance Jason in IRA Jason', 'Cumulative Inheritance Jason in IRA Elisa', \
                                    'Cumulative Inheritance Elisa in IRA Jason', 'Cumulative Inheritance Elisa in IRA Elisa', \
                                    'Cashflow Jason excess', 'Cumulative Cashflow Jason excess', \
                                    'Cashflow Portfolio Jason', 'Forward Gain Cumulative Jason', \
                                    'Value Jason', 'Cashflow Portfolio Elisa', 'Forward Gain Cumulative Elisa', 'Value Elisa']]

        value_present_Jason = values_Jason[values_Jason[col_gains_date] == date_before_first_outflow].iloc[0][col_values]
        value_present_Elisa = values_Elisa[values_Elisa[col_gains_date] == date_before_first_outflow].iloc[0][col_values]

        if fraction_Jason is None:
            share_Jason = value_present_Jason / (value_present_Jason + value_present_Elisa)
        else:
            share_Jason = fraction_Jason
        share_Elisa = 1 - share_Jason
    
        inheritance_Jason_in_IRA_Jason = inheritance_Jason_before_tax
        inheritance_Jason_in_IRA_Elisa = 0
        inheritance_Elisa_in_IRA_Jason = inheritance_Elisa_before_tax * share_Jason - inheritance_Jason_before_tax * share_Elisa
        inheritance_Elisa_in_IRA_Elisa = inheritance_Elisa_before_tax * share_Elisa + inheritance_Jason_before_tax * share_Elisa

        balance_before_first_outflow = values[values['Date'] == date_before_first_outflow]['Balance Bank'].values[0]

        total_inheritance_before_tax = inheritance_Jason_before_tax + inheritance_Elisa_before_tax
        total_inheritance_after_tax = total_inheritance_before_tax * (1. - tax_rate)
        date_of_first_inheritance_transfer = values[values['Balance Bank'] >= balance_before_first_outflow + total_inheritance_after_tax].iloc[-1]['Date']

        # Transfer assets from bank accounts to IRAs
        cumulative_inheritance = 0.
        exclusive_Jason_cumulative = 0.
        exclusive_Elisa_cumulative = 0.
        for index, row in values[(values['Date'] >= date_of_first_inheritance_transfer) & (values['Date'] < date_first_outflow)].iterrows():
            date = values.loc[index, 'Date']

            if index == 0:
                cashflow = -values.loc[index, 'Cashflow Bank'] / (1. - tax_rate)
            else:
                cashflow = min(total_inheritance_before_tax - cumulative_inheritance, -values.loc[index, 'Cashflow Bank'] / (1. - tax_rate))

            exclusive_Jason, exclusive_Elisa, share_Jason, share_Elisa, cashflow_Jason, cashflow_Elisa, delta = self.cashflows_calculate(date, values, exclusions, cashflow, tax_rate, fraction_Jason, debug=debug)

            inheritance_fraction_Jason = inheritance_Jason_before_tax / total_inheritance_before_tax
            inheritance_fraction_Elisa = inheritance_Elisa_before_tax / total_inheritance_before_tax

            inheritance_Jason_in_IRA_Jason = inheritance_fraction_Jason * cashflow
            inheritance_Jason_in_IRA_Elisa = 0
            inheritance_Elisa_in_IRA_Jason = inheritance_fraction_Elisa * cashflow_Jason - inheritance_fraction_Jason * cashflow_Elisa
            inheritance_Elisa_in_IRA_Elisa = inheritance_fraction_Elisa * cashflow_Elisa + inheritance_fraction_Jason * cashflow_Elisa

            cumulative_inheritance += cashflow

            values.loc[index, 'Inheritance Jason in IRA Jason'] = inheritance_Jason_in_IRA_Jason
            values.loc[index, 'Inheritance Jason in IRA Elisa'] = inheritance_Jason_in_IRA_Elisa
            values.loc[index, 'Inheritance Elisa in IRA Jason'] = inheritance_Elisa_in_IRA_Jason
            values.loc[index, 'Inheritance Elisa in IRA Elisa'] = inheritance_Elisa_in_IRA_Elisa

            values.loc[index, 'Share Jason'] = share_Jason
            values.loc[index, 'Share Elisa'] = share_Elisa
            values.loc[index, 'Cashflow Jason'] = cashflow_Jason
            values.loc[index, 'Cashflow Elisa'] = cashflow_Elisa
            values.loc[index, 'Exclusive Jason'] = exclusive_Jason
            values.loc[index, 'Exclusive Elisa'] = exclusive_Elisa
            values.loc[index, 'Delta'] = delta

        values['Cumulative Inheritance Jason in IRA Jason'] = values['Inheritance Jason in IRA Jason'].cumsum()
        values['Cumulative Inheritance Jason in IRA Elisa'] = values['Inheritance Jason in IRA Elisa'].cumsum()
        values['Cumulative Inheritance Elisa in IRA Jason'] = values['Inheritance Elisa in IRA Jason'].cumsum()
        values['Cumulative Inheritance Elisa in IRA Elisa'] = values['Inheritance Elisa in IRA Elisa'].cumsum()

        values['Cumulative Cashflow Jason'] = values['Cashflow Jason'].cumsum()
        values['Cumulative Cashflow Elisa'] = values['Cashflow Elisa'].cumsum()
        values['Cumulative Exclusive Jason'] = values['Exclusive Jason'].cumsum()
        values['Cumulative Exclusive Elisa'] = values['Exclusive Elisa'].cumsum()
        values['Cumulative Delta'] = values['Delta'].cumsum()

        # Transfer Elisa inheritance from Jason IRA to Elisa IRA, accumulate excess payments from Jason IRA
        for index, row in values[(values['Date'] >= date_first_outflow)].iterrows():
            date = values.loc[index, 'Date']

            cashflow = values.loc[index, 'Cashflow Portfolio Jason']
     
            exclusive_Jason, exclusive_Elisa, share_Jason, share_Elisa, cashflow_Jason, cashflow_Elisa, delta = self.cashflows_calculate(date, values, exclusions, cashflow, tax_rate, fraction_Jason, debug=debug)

            cashflow_inheritance = min(-cashflow_Elisa, values.loc[index, 'Cumulative Inheritance Elisa in IRA Jason'])
            values.loc[index, 'Inheritance Elisa in IRA Jason'] += -cashflow_inheritance
            values.loc[index, 'Inheritance Elisa in IRA Elisa'] += cashflow_inheritance

            if abs(values.loc[index, 'Cumulative Inheritance Elisa in IRA Jason'] - cashflow_inheritance) < 0.0001:
                cashflow_excess_Jason = -(values.loc[index, 'Cumulative Inheritance Elisa in IRA Jason'] + cashflow_Elisa) - delta
            else:
                cashflow_excess_Jason = -delta

            values.loc[index, 'Cashflow Jason excess'] = cashflow_excess_Jason
            values.loc[index:, 'Cumulative Inheritance Elisa in IRA Jason'] += -cashflow_inheritance
            values.loc[index:, 'Cumulative Inheritance Elisa in IRA Elisa'] += cashflow_inheritance

            values.loc[index, 'Share Jason'] = share_Jason
            values.loc[index, 'Share Elisa'] = share_Elisa
            values.loc[index, 'Cashflow Jason'] = cashflow_Jason
            values.loc[index, 'Cashflow Elisa'] = cashflow_Elisa
            values.loc[index, 'Exclusive Jason'] = exclusive_Jason
            values.loc[index, 'Exclusive Elisa'] = exclusive_Elisa
            values.loc[index, 'Delta'] = delta

        values['Cumulative Inheritance Jason in IRA Jason'] = values['Inheritance Jason in IRA Jason'].cumsum()
        values['Cumulative Inheritance Jason in IRA Elisa'] = values['Inheritance Jason in IRA Elisa'].cumsum()
        values['Cumulative Inheritance Elisa in IRA Jason'] = values['Inheritance Elisa in IRA Jason'].cumsum()
        values['Cumulative Inheritance Elisa in IRA Elisa'] = values['Inheritance Elisa in IRA Elisa'].cumsum()
        values['Cumulative Cashflow Jason excess'] = values['Cashflow Jason excess'].cumsum()

        values['Cumulative Cashflow Jason'] = values['Cashflow Jason'].cumsum()
        values['Cumulative Cashflow Elisa'] = values['Cashflow Elisa'].cumsum()
        values['Cumulative Exclusive Jason'] = values['Exclusive Jason'].cumsum()
        values['Cumulative Exclusive Elisa'] = values['Exclusive Elisa'].cumsum()
        values['Cumulative Delta'] = values['Delta'].cumsum()
        
        values['Cumulative Cashflow Portfolio Jason'] = values['Cashflow Portfolio Jason'].cumsum()
        values['Cumulative Cashflow Portfolio Elisa'] = values['Cashflow Portfolio Elisa'].cumsum()
        
        return values

    def cashflows_calculate(self, date, values, exclusions, cashflow, tax_rate, fraction_Jason, debug=0):
        value_present_Jason = values[values['Date'] == date].iloc[-1]['Value Jason']
        value_present_Elisa = values[values['Date'] == date].iloc[-1]['Value Elisa']

        if fraction_Jason is None:
            share_Jason = value_present_Jason / (value_present_Jason + value_present_Elisa)
        else:
            share_Jason = fraction_Jason
        share_Elisa = 1 - share_Jason
        
        mask_date = (self.splits['trn_date'] == date)
        df_splits = self.splits[mask_date]
        mask_true = (df_splits['trn_date'] == date) | True
        mask_false = (df_splits['trn_date'] == date) & False

        mask_outer = mask_false
        for exclusion_list in exclusions['Jason']:
            mask_inner = mask_true
            for exclusion in exclusion_list:
                if exclusion['comparison'] == 'equals':
                    mask_inner = mask_inner & (df_splits[exclusion['column']] == exclusion['value'])
                elif exclusion['comparison'] == 'contains':
                    mask_inner = mask_inner & (df_splits[exclusion['column']].str.contains(exclusion['value']))
            mask_outer = mask_outer | mask_inner

        df = df_splits[mask_outer].copy()
        exclusive_Jason = float(df['value'].sum()) / (1. - tax_rate)

        if (debug > 1) & (exclusive_Jason != 0):
            print(f"Date =  {date.date()}, exclusive_Jason = {exclusive_Jason:,.2f}")
            with pd.option_context('display.min_rows', None, 'display.max_rows', None, 'display.max_columns', None, 'display.float_format', '{:,.2f}'.format):
                display(df)

        mask_outer = mask_false
        for exclusion_list in exclusions['Elisa']:
            mask_inner = mask_true
            for exclusion in exclusion_list:
                if exclusion['comparison'] == 'equals':
                    mask_inner = mask_inner & (df_splits[exclusion['column']] == exclusion['value'])
                elif exclusion['comparison'] == 'contains':
                    mask_inner = mask_inner & (df_splits[exclusion['column']].str.contains(exclusion['value']))
            mask_outer = mask_outer | mask_inner

        df = df_splits[mask_outer].copy()
        exclusive_Elisa = float(df['value'].sum()) / (1. - tax_rate)

        if (debug > 1) & (exclusive_Elisa != 0):
            print(f"Date = {date.date()}, exclusive_Elisa = {exclusive_Elisa:,.2f}")
            with pd.option_context('display.min_rows', None, 'display.max_rows', None, 'display.max_columns', None, 'display.float_format', '{:,.2f}'.format):
                display(df)

        delta = exclusive_Jason * share_Elisa - exclusive_Elisa * share_Jason
        cashflow_Jason = cashflow * share_Jason
        cashflow_Elisa = cashflow * share_Elisa

        return exclusive_Jason, exclusive_Elisa, share_Jason, share_Elisa, cashflow_Jason, cashflow_Elisa, delta

    def exclusive_values(self, exclusions={}, date_beginning=None, date_ending=None):
        df_splits = self.splits.copy().sort_values(by=['trn_date'])
    
        if date_beginning is not None:
            mask_date = (df_splits['trn_date'] >= date_beginning)
        else:
            mask_date = (df_splits['trn_date'] >= 0)
    
        if date_ending is not None:
            mask_date = mask_date & (df_splits['trn_date'] <= date_ending)

        df_dates = df_splits[mask_date]

        mask_true = mask_date[mask_date] | True
        mask_false = mask_date[mask_date] & False

        mask_outer = mask_false
        for exclusion_list in exclusions['Jason']:
            mask_inner = mask_true
            for exclusion in exclusion_list:
                if exclusion['comparison'] == 'equals':
                    mask_inner = mask_inner & (df_dates[exclusion['column']] == exclusion['value'])
                elif exclusion['comparison'] == 'contains':
                    mask_inner = mask_inner & (df_dates[exclusion['column']].str.contains(exclusion['value']))
            mask_outer = mask_outer | mask_inner

        df_Jason = df_dates[mask_outer]
        df_Jason.loc[:, 'value'] = df_Jason['value'].astype(float)
        exclusive_Jason = float(df_Jason['value'].sum())

        mask_outer = mask_false
        for exclusion_list in exclusions['Elisa']:
            mask_inner = mask_true
            for exclusion in exclusion_list:
                if exclusion['comparison'] == 'equals':
                    mask_inner = mask_inner & (df_dates[exclusion['column']] == exclusion['value'])
                elif exclusion['comparison'] == 'contains':
                    mask_inner = mask_inner & (df_dates[exclusion['column']].str.contains(exclusion['value']))
            mask_outer = mask_outer | mask_inner

        df_Elisa = df_dates[mask_outer].copy()
        df_Elisa.loc[:, 'value'] = df_Elisa['value'].astype(float)
        exclusive_Elisa = float(df_Elisa['value'].sum())

        return exclusive_Jason, exclusive_Elisa, df_Jason, df_Elisa

    def print_exclusive_expenses(self, exclusions={}, date_beginning=None, date_ending=None, total_expenses=None):
        exclusive_Jason, exclusive_Elisa, df_Jason, df_Elisa = self.exclusive_values(exclusions, date_beginning=date_beginning, date_ending=date_ending)

        if total_expenses is None:
            print(f"After tax exclusive_Jason = {exclusive_Jason:,.2f}, Before tax exclusive_Jason = {exclusive_Jason / 0.8:,.2f}")
            print(f"After tax exclusive_Elisa = {exclusive_Elisa:,.2f}, Before tax exclusive_Elisa = {exclusive_Elisa / 0.8:,.2f}")
        else:
            print(f"After tax exclusive_Jason = {exclusive_Jason:,.2f}, Before tax exclusive_Jason = {exclusive_Jason / 0.8:,.2f}, fraction of total expenses = {exclusive_Jason / total_expenses:.1%}")
            print(f"After tax exclusive_Elisa = {exclusive_Elisa:,.2f}, Before tax exclusive_Elisa = {exclusive_Elisa / 0.8:,.2f}, fraction of total expenses = {exclusive_Elisa / total_expenses:.1%}")

        # if (exclusive_Jason + exclusive_Elisa) != 0: print(f"Sharing ratio, at which delta vanishes: share_Jason = {exclusive_Jason / (exclusive_Jason + exclusive_Elisa):.1%}, share_Elisa = {exclusive_Elisa / (exclusive_Jason + exclusive_Elisa):.1%}")

    def total_expenses(self, date_beginning=None, date_ending=None):
        exclusions_total = {}
        exclusions_total['Jason'] = []
        exclusions_total['Elisa'] = []

        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Auto', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Clothing', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Deposit', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Education', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Entertainment', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Expenses', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Fee', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Food & Grocery', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Gifts', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Government', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Grooming', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Health', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Household', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Imbalance-USD', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Insurance', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Interest', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Personal', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Professional', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Recreation', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Taxes', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Unknown', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Unspecified', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Utilities', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Vacation', 'comparison': 'contains'}])
        exclusions_total['Jason'].append([{'column': 'act_path', 'value': 'Wedding', 'comparison': 'contains'}])
        
        exclusive_total, exclusive_Elisa, df_total, df_Elisa = self.exclusive_values(exclusions_total, date_beginning=date_beginning, date_ending=date_ending)
        
        return exclusive_total, df_total


    style_format =  { \
                        'Date': '{:%Y-%m-%d}', \
                        'Cashflow Bank': '{:,.2f}', \
                        'Balance Bank': '{:,.2f}', \
                        'Share Jason': '{:.1%}', \
                        'Cashflow Jason': '{:,.2f}', \
                        'Cumulative Cashflow Jason': '{:,.2f}', \
                        'Exclusive Jason': '{:,.2f}', \
                        'Cumulative Exclusive Jason': '{:,.2f}', \
                        'Share Elisa': '{:.1%}', \
                        'Cashflow Elisa': '{:,.2f}', \
                        'Cumulative Cashflow Elisa': '{:,.2f}', \
                        'Exclusive Elisa': '{:,.2f}', \
                        'Cumulative Exclusive Elisa': '{:,.2f}', \
                        'Delta': '{:,.2f}', \
                        'Cumulative Delta': '{:,.2f}', \
                        'Cumulative Inheritance Jason in IRA Jason': '{:,.2f}', \
                        'Cumulative Inheritance Jason in IRA Elisa': '{:,.2f}', \
                        'Cumulative Inheritance Elisa in IRA Jason': '{:,.2f}', \
                        'Cumulative Inheritance Elisa in IRA Elisa': '{:,.2f}', \
                        'Cumulative Cashflow Jason excess': '{:,.2f}', \
                        'Cashflow Portfolio Jason': '{:,.2f}', \
                        'Cumulative Cashflow Portfolio Jason': '{:,.2f}', \
                        'Value Jason': '{:,.2f}', \
                        'Value Elisa': '{:,.2f}', \
                    }

def decimal_to_float(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if isinstance(df_copy[col].iloc[0], Decimal):
            df_copy[col] = df_copy[col].apply(pd.to_numeric, downcast='float')
    return df_copy
        
def display_format(df):
    df_copy = decimal_to_float(df)
    display(df_copy.style.format(format_transactions, thousands=',', precision=2))

# Display format for transaction DataFrames
format_transactions = {
    'Date': '{:%Y-%m-%d}',
}

def rate_calc_column(df, start=None, end=None, date_column='Date', cashflow_column='Cashflow', value_column='Value'):
    df['Forward Rate Daily'] = 0.
    df['Forward Rate Annual'] = 0.
    for index in df.index:
        start = df.loc[index, date_column]
        
        rate_daily = rate_calc_single(df, start=start, end=end, date_column=date_column, cashflow_column=cashflow_column, value_column=value_column)
        df.loc[index, 'Forward Rate Daily'] = rate_daily

        rate_annual = (1. + rate_daily)**365 - 1.
        df.loc[index, 'Forward Rate Annual'] = rate_annual

    return df

def rate_calc_single(df, start=None, end=None, date_column='Date', cashflow_column='Cashflow', value_column='Value'):
    df = df.copy()        
    root_results = optimize.root_scalar(delta_return, args=(df, start, end, date_column, cashflow_column, value_column), method='secant', x0=0., x1=0.0003)
    daily_gain = root_results.root
    return daily_gain
    
def delta_return(rate_daily, df, start=None, end=None, date_column='Date', cashflow_column='Cashflow', value_column='Value'):
    if start == None:
        start = df[date_column].min()
    if end == None:
        end = df[date_column].max()

    df_returns = df[(df[date_column] >= start) & (df[date_column] <= end)].copy().reset_index(drop=True)
    df_returns.loc[0, cashflow_column] = df_returns.loc[0, value_column]

    df_returns['Exponential Gain'] = (1 + rate_daily)**df_returns.index.values[::-1]
    df_returns['Discounted Cashflows'] = df_returns['Exponential Gain'] * df_returns[cashflow_column]
    df_returns['Discounted Cumulative Cashflows'] = df_returns['Discounted Cashflows'].cumsum()

    delta = df_returns['Discounted Cashflows'].sum() - df_returns[value_column].iloc[-1]

    return delta

# # Adapting Stefan's data structure to financial modeling
#
# from dataclasses import dataclass
#
# # class Porfolio_accounts  # data structure for portfolio choice, i.e. set of strings
# # class Book  # datastructure for market data
#
# @dataclass
# class Portfolio:
#     Porfolio_accounts: frozenset[str]
#     Book: book
#
#     def action(self):
#         return
        
import sys
import os
import yfinance as yf
import re
import pandas as pd

from datetime import datetime
from datetime import timedelta
from timeit import default_timer as timer
from io import StringIO
import time

# From https://github.com/ranaroussi/yfinance/blob/main/README.md
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter    

class Portfolio:
    """Builds portfolios from GnuCash XML Book.
    
       Filters the DataFrames in Book object, to create member DataFrames for the portfolio accounts and splits,
       based upon a subset of accounts in the Book.
    """
    
    def __init__(self, book, account_list, adjust_splits=True, calc_rates=False, debug=0):
        """
        Parameters
        ----------
        book :          Book object, constructed from GnuCash data file (XML format)
        account_list :  List of strings, consisting of names of accounts to be contained in the portfolio
                        Wildcards support:  Use 'Fidelity' to include the Fidelity account
                                            Use 'Fidelity:*' to include all children of Fidelity account
                                            Use 'Fidelity:**' to include all descendents of Fidelity account
        members :       Members include filtered accounts and splits DataFrames. If the portfolio does not exist in
                        the Book.portfolios dictionary, then the data objects are created, and a reference to the
                        Portfolio instance is included in the Book.portfolios dictionary.  If a portfolio with the 
                        specified accounts is already present in the Book.portfolios member, then a new instance
                        is created, and all existing data objects populated by reference.  
        """
        debug_threshold = 3

        if not isinstance(account_list, list): account_list = [account_list]
        key = frozenset(account_list)
        
        account_list_expanded = self.expand_account_list(book, account_list)
        ticker_dict_unsorted = self.get_ticker_dict(book, account_list)
        ticker_dict = collections.OrderedDict(sorted(ticker_dict_unsorted.items()))
        ticker_list = list(ticker_dict.values())
        
        self.excluded_equities = ['BBBYQ', 'ELX', 'FB', 'MSO', 'NYSE', 'NTOP', 'TIVO', 'LNUX', 'VIAC', 'WNR', 'WFM']
        equities_dict_unsorted = self.__fullHistories(ticker_list, period='max', adjust_splits=adjust_splits, debug=debug)
        
        equities_dict = collections.OrderedDict(sorted(equities_dict_unsorted.items()))
        self.equities_dict = equities_dict
        
        if ((key not in list(book.portfolios.keys())) & (account_list_expanded != [])):
            self.key = key
            self.book = book
            self.accounts = book.accounts[book.accounts['path'].isin(account_list_expanded)]
            self.splits = book.splits[book.splits['act_path'].isin(account_list_expanded)].sort_values(by=['trn_date', 'trn_id', 'act_path'])
            self.ticker_dict = ticker_dict
            self.__init_transactions(book, calc_rates=calc_rates, debug=debug)
            self.cashflows_init()

        else:
            self.key = book.portfolios[key].key
            self.book = book
            self.accounts = book.portfolios[key].accounts
            self.splits = book.portfolios[key].splits
            self.ticker_dict = book.portfolios[key].ticker_dict
            self.equities_dict = book.portfolios[key].equities_dict
            self.transactions = book.portfolios[key].transactions
            self.valuations = book.portfolios[key].valuations
            self.merged = book.portfolios[key].merged

        book.portfolios[key] = self

    def mask_internal_splits(self, internal_account_designators=['Income:', 'Expenses:']):
        mask = False
        for internal in internal_account_designators:
            mask = mask | self.performance['act_path'].str.contains(internal, na=False)

        return mask

    def mask_external_splits(self, internal_account_designators=['Income:', 'Expenses:']):
        mask = self.performance['act_path'].notnull()
        for internal in internal_account_designators:
            mask = mask & ~self.performance['act_path'].str.contains(internal, na=True)

        return mask

    def is_internal_account(self, account, internal_account_designators=['Income:', 'Expenses:']):
        is_internal = False
        for internal in internal_account_designators:
            is_internal = is_internal or (internal in account)
        
        return is_internal

    def is_external_account(self, account, internal_account_designators=['Income:', 'Expenses:']):
        
        return not self.is_internal_account(account, internal_account_designators)

    def __init_transactions(self, book, adjust_splits=True, internal_account_designators=['Income:', 'Expenses:'], calc_rates=False, debug=0):
        df_perf = self.splits[['trn_date', 'trn_id', 'act_path', 'trn_description', 'act_cmd_space', 'act_cmd_id', 'value', 'quantity']].copy().rename(columns={'trn_date': 'Date'})
        
        # Expand account list wildcards
        account_list_expanded = self.expand_account_list(book, list(self.key))
        account_list_expanded = sorted(account_list_expanded, key=lambda account: self.shortname(account))
        
        # Build DataFrame containing all transaction splits
        df_perf['value'] = df_perf['value'].astype(float)
        df_perf['Cashflow Portfolio'] = 0.
        df_perf['Cashflow Internal Portfolio'] = 0.
        df_perf['Value Portfolio'] = 0.
        
        for account in account_list_expanded:
            mask = df_perf['act_path'] == account
            shortname = self.shortname(account)
            
            df_perf.loc[mask, f'Cashflow {shortname}'] = df_perf[mask]['value'].astype(float)
            df_perf[f'Cashflow {shortname}'] = df_perf[f'Cashflow {shortname}'].fillna(0.)

            df_perf.loc[mask, 'Cashflow Portfolio'] = df_perf.loc[mask, 'Cashflow Portfolio'] + df_perf.loc[mask, f'Cashflow {shortname}']

            df_perf.loc[mask, f'Cashflow Cumulative {shortname}'] = df_perf[mask]['value'].cumsum().astype(float)
            df_perf[f'Cashflow Cumulative {shortname}'] = df_perf[f'Cashflow Cumulative {shortname}'].ffill().fillna(0.)

            # Build holdings, price, value and net return columns for investment accounts
            if self.is_external_account(account, internal_account_designators):
                df_perf.loc[mask, f'Holdings {shortname}'] = df_perf[mask]['quantity'].cumsum().astype(float)
                df_perf[f'Holdings {shortname}'] = df_perf[f'Holdings {shortname}'].ffill().fillna(0.)

                df_perf.loc[mask, f'Price {shortname}'] = df_perf[mask]['value'].astype(float) / df_perf[mask]['quantity'].astype(float)

                df_perf.loc[mask, f'Value {shortname}'] = df_perf.loc[mask, f'Holdings {shortname}'].astype(float) * df_perf.loc[mask, f'Price {shortname}'].astype(float)
                df_perf[f'Value {shortname}'] = df_perf[f'Value {shortname}'].ffill().fillna(0.)

                df_perf.loc[mask, f'Net {shortname}'] = df_perf.loc[mask, f'Value {shortname}'].astype(float) - df_perf.loc[mask, f'Cashflow Cumulative {shortname}'].astype(float)
                df_perf[f'Net {shortname}'] = df_perf[f'Net {shortname}'].ffill().fillna(0.)

                df_perf.loc[mask, 'Value Portfolio'] = df_perf.loc[mask, 'Value Portfolio'] + df_perf.loc[mask, f'Value {shortname}']
            else:
                df_perf.loc[mask, 'Cashflow Internal Portfolio'] = df_perf.loc[mask, 'Cashflow Internal Portfolio'] + df_perf.loc[mask, f'Cashflow {shortname}']

        df_perf['Cashflow Cumulative Portfolio'] = df_perf['Cashflow Portfolio'].cumsum().astype(float)
        df_perf['Cashflow Cumulative Internal Portfolio'] = df_perf['Cashflow Internal Portfolio'].cumsum().astype(float)

        df_perf = df_perf.reset_index()
        df_perf = self.columns_to_beginning( \
                    df_perf, \
                    [ \
                        'idtype', \
                        'id', \
                        'Date', \
                        'Cashflow Portfolio', \
                        'Cashflow Cumulative Portfolio', \
                        'Cashflow Internal Portfolio', \
                        'Cashflow Cumulative Internal Portfolio', \
                        'Value Portfolio', \
                        'trn_description', \
                        'act_path' \
                    ] \
                )
        
        # Build DataFrame containing all historical equity pricing
        date_start = df_perf['Date'].min()
        date_end = pd.to_datetime('today').normalize()

        dt_dates = pd.to_datetime(pd.date_range(start=date_start, end=date_end))
        df_daily_indexed = pd.DataFrame(index=dt_dates, columns=df_perf.columns, dtype=float)
        df_daily_indexed['trn_description'] = 'Valuation'
        df_daily_indexed['Date'] = dt_dates
        df_daily_indexed['idtype'] = 'uuid4'
        df_daily_indexed['id'] = df_daily_indexed.apply(lambda _: uuid.uuid4(), axis=1)
        
        for symbol in self.equities_dict.keys():
            account = self.account_from_ticker(self.book, symbol)
            shortname = self.shortname(account)
            account_history = self.equities_dict[symbol]
            daily_history = account_history.loc[dt_dates[0]:dt_dates[-1]].copy()

            df_daily_indexed.loc[daily_history.index, f"Price {shortname}"] = daily_history['Close']

        # Concatenate transaction splits with historical pricing
        df_merged = pd.concat([df_perf, df_daily_indexed])
        df_merged = df_merged.sort_values(by=['Date', 'idtype', 'trn_id', 'act_path'])
        df_merged.reset_index(drop=True, inplace=True)
        df_merged['Value Portfolio'] = 0.
        
        # Calculate cashflows, holdings, equity pricing, values and net returns for portfolio, daily, beginning with first transaction split
        for account in account_list_expanded:
            shortname = self.shortname(account)
            
            df_merged[f'Cashflow Cumulative {shortname}'] = df_merged[f'Cashflow Cumulative {shortname}'].ffill()
            df_merged[f'Cashflow Cumulative {shortname}'] = df_merged[f'Cashflow Cumulative {shortname}'].fillna(0.)
            df_merged[f'Cashflow {shortname}'] = df_merged[f'Cashflow {shortname}'].fillna(0.)
            if self.is_external_account(account, internal_account_designators):
                df_merged[f'Holdings {shortname}'] = df_merged[f'Holdings {shortname}'].ffill()
                df_merged[f'Price {shortname}'] = df_merged[f'Price {shortname}'].ffill()
                df_merged[f'Value {shortname}'] = df_merged[f'Holdings {shortname}'] * df_merged[f'Price {shortname}']
                df_merged[f'Net {shortname}'] = df_merged[f'Value {shortname}'] - df_merged[f'Cashflow Cumulative {shortname}']

                df_merged[f'Holdings {shortname}'] = df_merged[f'Holdings {shortname}'].fillna(0.)
                df_merged[f'Price {shortname}'] = df_merged[f'Price {shortname}'].fillna(0.)
                df_merged[f'Value {shortname}'] = df_merged[f'Value {shortname}'].fillna(0.)
                df_merged[f'Net {shortname}'] = df_merged[f'Net {shortname}'].fillna(0.)

                df_merged['Value Portfolio'] = df_merged['Value Portfolio'] + df_merged[f'Value {shortname}']
            else:
                df_merged[f'Net {shortname}'] = -df_merged[f'Cashflow Cumulative {shortname}']

        # For cashflow columns, fill each NA value with 0.
        investment_columns_noncumulative = [c for c in list(df_merged.columns) if ('Cashflow' in c) & ('Cashflow Cumulative' not in c)]
        for column in investment_columns_noncumulative:
            df_merged[column] = df_merged[column].fillna(0.)
        
        # For accumulating columns, fill each NA value with most recent value
        investment_columns_cumulative = [ \
            c for c in list(df_merged.columns) if \
                ('Holding' in c) \
                | ('Value' in c) \
                | ('Price' in c) \
                | ('Net' in c) \
                | ('Return' in c) \
                | ('Cashflow Cumulative' in c)\
        ]
        for column in investment_columns_cumulative:
            df_merged[column] = df_merged[column].ffill()
            
        # Calculate net portfolio returns
        df_merged['Net Portfolio'] = df_merged['Value Portfolio'] - df_merged['Cashflow Cumulative Portfolio']

        df_merged = self.columns_to_beginning( \
                        df_merged, \
                        [ \
                            'idtype', \
                            'id', \
                            'Date', \
                            'Cashflow Portfolio', \
                            'Cashflow Cumulative Portfolio', \
                            'Cashflow Internal Portfolio', \
                            'Cashflow Cumulative Internal Portfolio', \
                            'Value Portfolio', \
                            'Net Portfolio', \
                            'Forward Rate Daily', \
                            'Forward Gain Cumulative', \
                            'Forward Rate Annual', \
                            'trn_description', \
                            'act_path' \
                        ] \
                    )

        df_merged.loc[0, 'Cashflow Portfolio'] = df_merged.loc[0, 'Cashflow Cumulative Portfolio']
        df_merged.loc[0, 'Cashflow Internal Portfolio'] = 0.
        digits = 6
        df_merged.loc[1:, 'Cashflow Portfolio'] = (df_merged['Cashflow Cumulative Portfolio'][1:].values - df_merged['Cashflow Cumulative Portfolio'][0:-1].values).round(decimals=digits)
        df_merged.loc[1:, 'Cashflow Internal Portfolio'] = (df_merged['Cashflow Cumulative Internal Portfolio'][1:].values - df_merged['Cashflow Cumulative Internal Portfolio'][0:-1].values).round(decimals=digits)

        df_merged = df_merged.set_index('id')
        df_merged = df_merged.reset_index()
        df_merged = df_merged.sort_values(by=['Date', 'idtype', 'trn_id', 'act_path'])
        
        df_transactions = df_merged[df_merged['idtype'] == 'guid'].reset_index(drop=True)

        df_valuations = df_merged[df_merged['idtype'] == 'uuid4'].drop(columns=['trn_description', 'act_path', 'trn_id', 'act_cmd_space', 'act_cmd_id', 'quantity', 'value'])
        df_valuations = df_valuations.reset_index(drop=True)

        digits = 6
        for account in account_list_expanded:
            shortname = self.shortname(account)
            df_valuations.loc[0:, f'Cashflow {shortname}'] = df_valuations[f'Cashflow Cumulative {shortname}'][0:].values
            df_valuations.loc[1:, f'Cashflow {shortname}'] = (df_valuations[f'Cashflow Cumulative {shortname}'][1:].values - df_valuations[f'Cashflow Cumulative {shortname}'][0:-1].values).round(decimals=digits)

        df_valuations.loc[0, 'Cashflow Portfolio'] = df_valuations.loc[0, 'Value Portfolio']
        df_valuations.loc[1:, f'Cashflow Portfolio'] = (df_valuations[f'Cashflow Cumulative Portfolio'][1:].values - df_valuations[f'Cashflow Cumulative Portfolio'][0:-1].values).round(decimals=digits)

        # Calculate Forwrd Gain Daily
        df_valuations.loc[0, 'Forward Gain Daily'] = 1.
        df_valuations.loc[1:, 'Forward Gain Daily'] = \
            (df_valuations['Value Portfolio'][1:].values - df_valuations['Cashflow Portfolio'][1:].values - df_valuations['Cashflow Internal Portfolio'][1:].values) \
            / df_valuations['Value Portfolio'][0:-1].values

        df_valuations['Forward Gain Cumulative'] =  df_valuations['Forward Gain Daily'].cumprod()
        
        if calc_rates:
            start = np.datetime64('now')
            rate_calc_column(df_valuations, start=None, end=None, date_column='Date', cashflow_column='Cashflow Portfolio', value_column='Value Portfolio')
            end = np.datetime64('now')
            elapsed = end - start

        df_valuations = self.columns_to_beginning( \
                        df_valuations, \
                        [ \
                            'idtype', \
                            'id', \
                            'Date', \
                            'Cashflow Portfolio', \
                            'Cashflow Cumulative Portfolio', \
                            'Cashflow Internal Portfolio', \
                            'Cashflow Cumulative Internal Portfolio', \
                            'Value Portfolio', \
                            'Net Portfolio', \
                            'Forward Gain Daily', \
                            'Forward Gain Cumulative', \
                            'Forward Rate Daily', \
                            'Forward Rate Annual', \
                        ] \
                    )

        self.merged = df_merged
        self.valuations = df_valuations
        self.transactions = df_transactions

        return df_transactions
        
    def expand_account_list(self, book, account_list):
        account_list_expanded = []
        for account_string in account_list:
            if account_string[-3:] == ':**':
                accounts_string = self.__account_descendents(book, account_string[:-3])
                for account_descendent in accounts_string:
                    account_list_expanded.append(account_descendent)
            elif account_string[-2:] == ':*':
                accounts_string = self.__account_children(book, account_string[:-2])
                for account_child in accounts_string:
                    account_list_expanded.append(account_child)
            else:
                account_list_expanded.append(account_string)
                
        return account_list_expanded

    def get_ticker_list(self, book, account_list):
        account_list_expanded = self.expand_account_list(book, account_list)
        ticker_list = []
        for account in account_list_expanded:
            df = book.accounts[book.accounts['path'] == account]
            if df['cmd_space'].iloc[0] == 'NONCURRENCY':
                ticker = df['cmd_id'].iloc[0]
                if ticker == 'BRKB': ticker = 'BRK-B'
                ticker_list.append(ticker)
                
        return ticker_list
        
    def get_ticker_dict(self, book, account_list):
        account_list_expanded = self.expand_account_list(book, account_list)
        ticker_dict = {}
        for account in account_list_expanded:
            df = book.accounts[book.accounts['path'] == account]
            if df['cmd_space'].iloc[0] == 'NONCURRENCY':
                ticker = df['cmd_id'].iloc[0]
                if ticker == 'BRKB': ticker = 'BRK-B'
                ticker_dict[account] = ticker
                
        return ticker_dict
        
    def __account_children(self, book, account):
        """
        Return a list of paths for children of specified account
        Parameters
        ----------
        book :      Book object, constructed from GnuCash data file (XML format)
        account :   Account path; return list will contain children, but not the specified account
        """
        if account == '':
            return book.accounts[(book.accounts['parent_path'].isnull()) & (book.accounts['path'].notnull())]['path'].values.tolist()
        else:
            return book.accounts[(book.accounts['parent_path'].notnull()) & (book.accounts['parent_path']==account)]['path'].values.tolist()

    def __account_descendents(self, book, account):
        """
        Return a list of paths for descendents of specified account
        Parameters
        ----------
        book :      Book object, constructed from GnuCash data file (XML format)
        account :   Account path; return list will contain descendents, but not the specified account
        """
        if account == '':
            return book.accounts[book.accounts['path'].notnull()]['path'].values.tolist()
        else:
            return book.accounts[(book.accounts['path'].notnull()) & (book.accounts['path'].str.contains(account+":"))]['path'].values.tolist()

    def __is_equity(self, book, account):
        equity = book.accounts[book.accounts['path'] == account]['cmd_space'].iloc[0] == 'NONCURRENCY'
        return equity
        
    def ticker_from_account(self, book, account):
        if self.__is_equity(book, account): return self.ticker_dict[account]
        return account
        
    def account_from_ticker(self, book, ticker):
        try:
            key = next(key for key, value in self.ticker_dict.items() if value == ticker)
            return key
        except:
            print(f"ticker symbol {ticker} not found")
            return None

    def columns_to_end(self, df, columns):
        df = df[[c for c in df if c not in columns] + [c for c in columns if c in df]]
        return df

    def columns_to_beginning(self, df, columns):
        df = df[[c for c in columns if c in df] + [c for c in df if c not in columns]]
        return df

    class __Capturing(list):
        def __enter__(self):
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StringIO()
            return self
        def __exit__(self, *args):
            self.extend(self._stringio.getvalue().splitlines())
            del self._stringio    # free up some memory
            sys.stdout = self._stdout

    def __eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    class __CachedLimiterSession(CacheMixin, LimiterMixin, Session):
        pass

    def __fullHistories(self, equities, period='max', adjust_splits=True, debug=0):
        debug_threshold = 2

        if not isinstance(equities, list):
            equities = [equities]

        # Removed after update to yfinance package
        # session = self.__CachedLimiterSession(
        #     limiter=Limiter(RequestRate(2, Duration.SECOND)),  # max 2 requests per 5 seconds
        #     bucket_class=MemoryQueueBucket,
        #     backend=SQLiteCache("yfinance.cache"),
        #     expire_after=timedelta(minutes=3)
        # )

        # session.headers['User-agent'] = 'historical_finance/1.0'

        equities_dict = {}
        
        for symbol in equities:
            if symbol not in self.excluded_equities:
                # df = self.__saveEquity(session, symbol=symbol, period=period, adjust_splits=adjust_splits, debug=debug)
                df = self.__saveEquity(symbol=symbol, period=period, adjust_splits=adjust_splits, debug=debug)
                equities_dict[symbol] = df
            
        # os.remove("yfinance.cache")

        return equities_dict

    def __saveEquity(self, symbol='NVDA', period='max', interval='1d', adjust_splits=True, debug=0):
        debug_threshold = 2
        
        maxAttempts = 5

        # ticker = yf.Ticker(symbol, session=session)
        ticker = yf.Ticker(symbol)
        filename = './data/equities/' + symbol.replace('^', '_') + '.csv'

        with self.__Capturing() as output:
            attempt = 0
            while attempt<maxAttempts:
                attempt += 1
                try:
                    symbolError = False
                    df = pd.DataFrame()
                    df = ticker.history(period=period, interval=interval, auto_adjust=False)
                    break
                except:
                    symbolError = True
                    time.sleep(5 * attempt)

        downloadError = (len(output)>0) | (len(df)<=1)
        if (downloadError):
            pass
        elif not symbolError:
            df = df.reset_index()
            df['Date'] = df['Date'].dt.tz_localize(None)
            df = df.set_index('Date')
            
            df.to_csv(filename)

            if adjust_splits:
                stock_splits = df.copy()
                stock_splits.loc[stock_splits['Stock Splits'] == 0, 'Stock Splits'] = 1
                prod_splits = stock_splits['Stock Splits'].cumprod().iloc[-1]
                stock_splits.loc[:, 'Stock Splits, Cumulative'] = prod_splits / stock_splits['Stock Splits'].cumprod()
                df.loc[:,["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].multiply(stock_splits['Stock Splits, Cumulative'], axis="index")

            digits = 4
            df.loc[:,["Open"]] = df[["Open"]].round(digits)
            df.loc[:,["High"]] = df[["High"]].round(digits)
            df.loc[:,["Low"]] = df[["Low"]].round(digits)
            df.loc[:,["Close"]] = df[["Close"]].round(digits)
            df.loc[:,["Adj Close"]] = df[["Adj Close"]].round(digits)

        return df
    
    def shortname(self, account):
        ticker_dict = {}
        for key in self.ticker_dict.keys():
            ticker_dict[key.split(':')[-1]] = self.ticker_dict[key]
        
        shortname = ''
        for key in ticker_dict.keys():
            if key in account:
                shortname = ticker_dict[key]
            
        if 'Jason Elisa' in account:
            if shortname != '': shortname = ':' + shortname
            shortname = 'Jason & Elisa' + shortname
        elif 'Jason' in account:
            if shortname != '': shortname = ':' + shortname
            shortname = 'Jason' + shortname
        elif 'Elisa' in account:
            if shortname != '': shortname = ':' + shortname
            shortname = 'Elisa' + shortname

        if 'Checking' in account:
            if shortname != '': shortname = ':' + shortname
            shortname = 'Checking' + shortname
            
        if 'Fidelity' in account:
            if shortname != '': shortname = ':' + shortname
            shortname = 'Fidelity' + shortname
        elif 'Wells Fargo' in account:
            if shortname != '': shortname = ':' + shortname
            shortname = 'Wells Fargo' + shortname
            
        if 'Dividend' in account:
            if shortname != '': shortname = ':' + shortname
            shortname = 'Dividend' + shortname
            
        if 'Commission' in account:
            if shortname != '': shortname = ':' + shortname
            shortname = 'Commission' + shortname
            
        if shortname == '':
            shortname = account

        return shortname

    def date_delta(self, date, years=0, months=0, days=0):
        date_pd = pd.to_datetime(date)
        date_days = date_pd + pd.Timedelta(days, 'd')
    
        day = date_days.day

        month = int((date_days.month + months - 1) % 12) + 1
        year_carry = int((date_days.month + months - month) / 12)
    
        year = date_days.year + years + year_carry

        date_return = pd.to_datetime(dt.date(year, month, day))
        return date_return

    def print_portfolio_return(self, date, date_final=None, include_SPY=True):
        if date_final is None:
            date_current_np = np.datetime64('today')
            date_current_pd = pd.to_datetime(date_current_np)
            valuations = self.valuations.copy()
        else:
            date_current_pd = date_final
            date_current_np = date_final.to_numpy()
            mask_portfolio = (self.valuations['Date'] <= date_final)
            valuations = self.valuations[mask_portfolio].copy()

        date_start = valuations.iloc[0]['Date']
        days = (date_current_pd - date).days
        years = days / 365.25

        if date >= date_start:
            rate = valuations[valuations['Date'] == date]['Forward Rate Annual'].values[0]
            # gain = (1 + rate)**years
            gain_end = valuations[valuations['Date'] <= date_current_np]['Forward Gain Cumulative'].values[-1]
            gain_start = valuations[valuations['Date'] == date]['Forward Gain Cumulative'].values[0]
            gain = gain_end / gain_start
            
            if include_SPY:
                mask = self.equities_dict['SPY'].index <= date
                close_SPY_current = self.equities_dict['SPY'].iloc[-1]['Adj Close']
                close_SPY_prior = self.equities_dict['SPY'][mask].iloc[-1]['Adj Close']
                rate_SPY = (close_SPY_current / close_SPY_prior)**(1/years) - 1.
                gain_SPY = (1 + rate_SPY)**years

                if years < 0.98:
                    print(f"days = {days:>5.0f}, starting date = {date.date()}, portfolio APR = {rate:>5.1%}, SPY APR = {rate_SPY:>5.1%}, portfolio gain = {gain:>5.2f}, SPY gain = {gain_SPY:>5.2f}")
                else:
                    years_frac = (years - round(years,0))
                    if abs(years_frac) < 0.05:
                        print(f"years = {years:>4.0f}, starting date = {date.date()}, portfolio APR = {rate:>5.1%}, SPY APR = {rate_SPY:>5.1%}, portfolio gain = {gain:>5.2f}, SPY gain = {gain_SPY:>5.2f}")
                    else:
                        print(f"years = {years:>4.1f}, starting date = {date.date()}, portfolio APR = {rate:>5.1%}, SPY APR = {rate_SPY:>5.1%}, portfolio gain = {gain:>5.2f}, SPY gain = {gain_SPY:>5.2f}")
            else:
                if years < 0.98:
                    print(f"days = {days:>5.0f}, starting date = {date.date()}, portfolio APR = {rate:>5.1%}, portfolio gain = {gain:>5.2f}")
                else:
                    years_frac = (years - round(years,0))
                    if abs(years_frac) < 0.05:
                        print(f"years = {years:>4.0f}, starting date = {date.date()}, portfolio APR = {rate:>5.1%}, portfolio gain = {gain:>4.1f}")
                    else:
                        print(f"years = {years:>4.1f}, starting date = {date.date()}, portfolio APR = {rate:>5.1%}, portfolio gain = {gain:>4.1f}")

    def print_portfolio_returns(self, date_final=None, include_SPY=True):
        if date_final is None:
            date_current_np = np.datetime64('today')
            date_current_pd = pd.to_datetime(date_current_np)
            valuations = self.valuations.copy()
        else:
            date_current_pd = date_final
            date_current_np = date_final.to_numpy()
            mask_portfolio = (self.valuations['Date'] <= date_final)
            valuations = self.valuations[mask_portfolio].copy()

        date_start = self.valuations.iloc[0]['Date']
    
        date = pd.to_datetime(dt.date(date_current_pd.year, 1, 1))
        self.print_portfolio_return(date, date_final=date_final, include_SPY=include_SPY)
        
        years = 1
        date = self.date_delta(date_current_np, years=-years)
        self.print_portfolio_return(date, date_final=date_final, include_SPY=include_SPY)

        years = 2
        date = self.date_delta(date_current_np, years=-years)
        self.print_portfolio_return(date, date_final=date_final, include_SPY=include_SPY)

        years = 5
        date = self.date_delta(date_current_np, years=-years)
        self.print_portfolio_return(date, date_final=date_final, include_SPY=include_SPY)

        years = 10
        date = self.date_delta(date_current_np, years=-years)
        self.print_portfolio_return(date, date_final=date_final, include_SPY=include_SPY)

        years = 20
        date = self.date_delta(date_current_np, years=-years)
        self.print_portfolio_return(date, date_final=date_final, include_SPY=include_SPY)

        date = self.valuations.iloc[0]['Date']
        days = (date_current_pd - date).days
        self.print_portfolio_return(date, date_final=date_final, include_SPY=include_SPY)
        
    def print_yearly_values(self, date_final=None):
        if date_final is None:
            mask_portfolio = (self.valuations['Date'] <= pd.Timestamp.now()) | True
            date_current_np = np.datetime64('today')
            date_current_pd = pd.to_datetime(date_current_np)
        else:
            mask_portfolio = (self.valuations['Date'] <= date_final)
            date_current_np = date_final.to_numpy()

        columns = [c for c in self.valuations.columns if ('trn_id' not in c) & ('Net' in c) | ('Date' in c) | ('Value Portfolio' in c) | ('Forward Gain Cumulative' in c) \
                    | ('Holdings' in c) | ('Price' in c) | ('Value' in c) \
                  ]
        valuations = self.valuations[mask_portfolio][columns].copy()

        date_first = valuations.iloc[0]['Date']
        date_last = valuations.iloc[-1]['Date']
        mask = (valuations['Date'] == date_last)

        year_first = date_first.year
        year_last = date_last.year

        for years in range(year_last - year_first + 1):
            date = self.date_delta(date_current_np, years=-years)
            if date >= date_first:
                mask = mask | (valuations['Date'] == date)

        valuations = valuations[mask][columns].copy()
        valuations['Value Portfolio, annual'] = valuations['Value Portfolio'] - valuations['Value Portfolio'].shift(1)
        valuations['Net Portfolio, annual'] = valuations['Net Portfolio'] - valuations['Net Portfolio'].shift(1)
        valuations['Cashflow Portfolio, annual'] = valuations['Value Portfolio, annual'] - valuations['Net Portfolio, annual']
        valuations.fillna(value=0., inplace=True)
        valuations = self.columns_to_beginning(valuations, ['Date', 'Value Portfolio', 'Net Portfolio', 'Value Portfolio, annual', 'Cashflow Portfolio, annual', 'Net Portfolio, annual'])

        with pd.option_context('display.max_columns', None, 'display.float_format', '{:,.2f}'.format):
            display(valuations)

    def print_portfolio_values(self, lines=1, date_begin=pd.Timestamp(2019,7,8), date_final=None):
        if date_final is None:
            mask_portfolio = (self.valuations['Date'] <= pd.Timestamp.now()) | True
            date_current_np = np.datetime64('today')
            date_current_pd = pd.to_datetime(date_current_np)
        else:
            mask_portfolio = (self.valuations['Date'] <= date_final)
            date_current_pd = date_final
            date_current_np = date_final.to_numpy()

        valuations = self.valuations[mask_portfolio].copy()

        # date_ytd = pd.to_datetime(dt.date(date_current_pd.year, 1, 1))
        # date_1 = self.date_delta(date_current_np, years=-1)
        # date_2 = self.date_delta(date_current_np, years=-2)
        # date_5 = self.date_delta(date_current_np, years=-5)
        # date_10 = self.date_delta(date_current_np, years=-10)
        # date_20 = self.date_delta(date_current_np, years=-20)
        # date_first = self.valuations.iloc[0]['Date']

        date_cashflows_begin = pd.Timestamp(2020,3,1)
        date_cashflows_end = pd.Timestamp(2021,4,1)

        value_max = valuations['Value Portfolio'].max()
        date_value_max = valuations[valuations['Value Portfolio'] == value_max]['Date'].iloc[-1]

        net_max = valuations['Net Portfolio'].max()
        date_net_max = valuations[valuations['Net Portfolio'] == net_max]['Date'].iloc[-1]

        gain_max = valuations['Forward Gain Cumulative'].max()
        date_gain_max = valuations[valuations['Forward Gain Cumulative'] == gain_max]['Date'].iloc[-1]

        columns = [c for c in valuations.columns[2:] if ('trn_id' not in c) & ('Net' in c) | ('Date' in c) | ('Value Portfolio' in c) | ('Forward Gain Cumulative' in c) \
                    | ('Holdings' in c) | ('Price' in c) | ('Value' in c) \
                  ]
        mask = (valuations.index >= valuations.index[-lines]) | (valuations['Date'] == date_begin) | (valuations['Date'] == date_value_max) \
                | (valuations['Date'] == date_net_max) | (valuations['Date'] == date_gain_max) | (valuations['Date'] == date_cashflows_begin) | (valuations['Date'] == date_cashflows_end) \
                # | (valuations['Date'] == date_ytd)| (valuations['Date'] == date_1)| (valuations['Date'] == date_2)| (valuations['Date'] == date_5)| (valuations['Date'] == date_10) \
                # | (valuations['Date'] == date_20)| (valuations['Date'] == date_first)

        with pd.option_context('display.max_columns', None, 'display.float_format', '{:,.2f}'.format):
            display(valuations[mask][columns])

    def print_net_max(self, date_final=None):
        if date_final is None:
            mask_portfolio = (self.valuations['Date'] <= pd.Timestamp.now()) | True
        else:
            mask_portfolio = (self.valuations['Date'] <= date_final)
        valuations = self.valuations[mask_portfolio].copy()

        date_start = valuations['Date'].iloc[0].date()

        net_max = valuations['Net Portfolio'].max()
        date_net_max = valuations[valuations['Net Portfolio'] == net_max]['Date'].iloc[-1].date()
        print(f"Net Portfolio maximum on {date_net_max}, with net value of ${net_max:,.2f}")
        
        gain_max = valuations['Forward Gain Cumulative'].max()
        date_gain_max = valuations[valuations['Forward Gain Cumulative'] == gain_max]['Date'].iloc[-1].date()
        print(f"Forward Gain Cumulative, from {date_start}, maximum on {date_gain_max}, with gain of {gain_max:,.2f}")
    
    def print_value_max(self, date_final=None):
        if date_final is None:
            mask_portfolio = (self.valuations['Date'] <= pd.Timestamp.now()) | True
        else:
            mask_portfolio = (self.valuations['Date'] <= date_final)
        valuations = self.valuations[mask_portfolio].copy()

        date_start = valuations['Date'].iloc[0].date()

        value_max = valuations['Value Portfolio'].max()
        date_value_max = valuations[valuations['Value Portfolio'] == value_max]['Date'].iloc[-1].date()
        print(f"Value Portfolio maximum on {date_value_max}, with value of ${value_max:,.2f}")
        
    def print_ending_net_values(self, lines=1, date_begin=pd.Timestamp(2019,7,8)):
        columns = [c for c in self.valuations.columns[2:] if ('trn_id' not in c) & ('Net' in c) | ('Date' in c) | ('Value Portfolio' in c)]
        mask = (self.valuations.index >= self.valuations.index[-lines]) | (self.valuations['Date'] == date_begin)
        with pd.option_context('display.max_columns', None, 'display.float_format', '{:,.2f}'.format):
            display(self.valuations[mask][columns])

    def print_ending_valuations(self, lines=1, date_begin=pd.Timestamp(2019,7,8)):
        columns = [c for c in self.valuations.columns[2:] if ('trn_id' not in c) & ('Date' in c) | ('Value Portfolio' in c) | ('Cashflow Cumulative Portfolio' in c) | ('Holdings' in c) | ('Price' in c) | ('Value' in c) | ('Forward Gain Cumulative' in c)]
        mask = (self.valuations.index >= self.valuations.index[-lines]) | (self.valuations['Date'] == date_begin)
        with pd.option_context('display.max_columns', None, 'display.float_format', '{:,.2f}'.format):
            display(self.valuations[mask][columns])

    def list_depth(lst):
        return isinstance(lst, list) and max(map(list_depth, lst)) + 1

    def wrap_list(self):
        if self.list_depth(cashflows) < 2:
            cashflows = [cashflows]

    def cashflows_init(self, debug=0):
        digits = 4
        
        cashflows_portfolio = self.valuations[['Date', 'Cashflow Portfolio']].copy()
        cashflows_portfolio = cashflows_portfolio[cashflows_portfolio['Cashflow Portfolio'].round(digits) != 0.]
        cashflows_portfolio = cashflows_portfolio.round(digits)
        cashflows_portfolio = cashflows_portfolio.reset_index(drop=True)
        self.cashflows_portfolio = cashflows_portfolio

    def cashflows_no_bank_init(self, debug=0):
        digits = 4
        
        self.tax_rate_fed = 0.15
        self.tax_rate_state = 0.03
        self.withdrawal_penalty = 0.1

        months = 14
        months_penalties = 9

        rf = self.tax_rate_fed
        rs = self.tax_rate_state
        rp = self.withdrawal_penalty

        self.inheritance_after_tax = 37500.00 + 37500.00 + 16500.00 + 13750.00 + 5341.27 + 412.27
        self.inheritance_before_tax = self.inheritance_after_tax / (1 - rf - rs)

        cash_monthly_inheritance = self.inheritance_after_tax / months

        cash_monthly_before_tax = -cash_monthly_inheritance / (1 - rf - rs)
        cash_monthly_before_tax_penalty = -cash_monthly_inheritance / (1 - rf - rs - rp)

        self.cashflow_before_tax = cash_monthly_before_tax
        cash_monthly_federal = cash_monthly_before_tax * rf
        cash_monthly_state = cash_monthly_before_tax * rs

        self.cashflow_before_tax_penalty = cash_monthly_before_tax_penalty
        cash_monthly_federal_penalty = cash_monthly_before_tax_penalty * rf
        cash_monthly_state_penalty = cash_monthly_before_tax_penalty * rs
        cash_monthly_penalty = cash_monthly_before_tax_penalty * rp

        self.cashflows_inheritance_total = self.cashflow_before_tax_penalty * months_penalties + self.cashflow_before_tax * (months - months_penalties)
        
        cashflows_IRA_no_bank_list = [
            [pd.to_datetime(dt.date(2020, 3, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2020, 3, 1)), cash_monthly_federal_penalty, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2020, 3, 1)), cash_monthly_state_penalty, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2020, 3, 1)), cash_monthly_penalty, 'Distribution Early Withdrawal Penalty'],
            [pd.to_datetime(dt.date(2020, 4, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2020, 4, 1)), cash_monthly_federal_penalty, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2020, 4, 1)), cash_monthly_state_penalty, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2020, 4, 1)), cash_monthly_penalty, 'Distribution Early Withdrawal Penalty'],
            [pd.to_datetime(dt.date(2020, 5, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2020, 5, 1)), cash_monthly_federal_penalty, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2020, 5, 1)), cash_monthly_state_penalty, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2020, 5, 1)), cash_monthly_penalty, 'Distribution Early Withdrawal Penalty'],
            [pd.to_datetime(dt.date(2020, 6, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2020, 6, 1)), cash_monthly_federal_penalty, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2020, 6, 1)), cash_monthly_state_penalty, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2020, 6, 1)), cash_monthly_penalty, 'Distribution Early Withdrawal Penalty'],
            [pd.to_datetime(dt.date(2020, 7, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2020, 7, 1)), cash_monthly_federal_penalty, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2020, 7, 1)), cash_monthly_state_penalty, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2020, 7, 1)), cash_monthly_penalty, 'Distribution Early Withdrawal Penalty'],
            [pd.to_datetime(dt.date(2020, 8, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2020, 8, 1)), cash_monthly_federal_penalty, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2020, 8, 1)), cash_monthly_state_penalty, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2020, 8, 1)), cash_monthly_penalty, 'Distribution Early Withdrawal Penalty'],
            [pd.to_datetime(dt.date(2020, 9, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2020, 9, 1)), cash_monthly_federal_penalty, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2020, 9, 1)), cash_monthly_state_penalty, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2020, 9, 1)), cash_monthly_penalty, 'Distribution Early Withdrawal Penalty'],
            [pd.to_datetime(dt.date(2020, 10, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2020, 10, 1)), cash_monthly_federal_penalty, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2020, 10, 1)), cash_monthly_state_penalty, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2020, 10, 1)), cash_monthly_penalty, 'Distribution Early Withdrawal Penalty'],
            [pd.to_datetime(dt.date(2020, 11, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2020, 11, 1)), cash_monthly_federal_penalty, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2020, 11, 1)), cash_monthly_state_penalty, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2020, 11, 1)), cash_monthly_penalty, 'Distribution Early Withdrawal Penalty'],
            [pd.to_datetime(dt.date(2020, 12, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2020, 12, 1)), cash_monthly_federal, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2020, 12, 1)), cash_monthly_state, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2021, 1, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2021, 1, 1)), cash_monthly_federal, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2021, 1, 1)), cash_monthly_state, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2021, 2, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2021, 2, 1)), cash_monthly_federal, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2021, 2, 1)), cash_monthly_state, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2021, 3, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2021, 3, 1)), cash_monthly_federal, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2021, 3, 1)), cash_monthly_state, 'Distribution Tax State'],
            [pd.to_datetime(dt.date(2021, 4, 1)), -cash_monthly_inheritance, 'Distribution Cash Flow'],
            [pd.to_datetime(dt.date(2021, 4, 1)), cash_monthly_federal, 'Distribution Tax Federal'],
            [pd.to_datetime(dt.date(2021, 4, 1)), cash_monthly_state, 'Distribution Tax State'],
        ]
        df_cashflows = pd.DataFrame(cashflows_IRA_no_bank_list, columns=['Date', 'Cashflow Portfolio', 'Description'])
        self.cashflows_IRA_no_bank_distributions = df_cashflows
        df_cashflows = df_cashflows[['Date', 'Cashflow Portfolio']]
        
        df_cashflows = pd.concat([self.cashflows_portfolio, df_cashflows], ignore_index=True).copy()
        df_cashflows = df_cashflows.groupby(['Date'], as_index=False).aggregate({'Cashflow Portfolio': 'sum'})
        df_cashflows = df_cashflows.sort_values(by=['Date'])
        df_cashflows = df_cashflows.round(digits)
        df_cashflows = df_cashflows.reset_index(drop=True)
        self.cashflows_IRA_no_bank = df_cashflows

        date_end = pd.to_datetime('today').normalize()
        cashflows_IRA_bank_list = [
            [date_end, -cash_monthly_inheritance * months, 'Distribution Cash Flow'],
            [date_end, cash_monthly_federal_penalty * months, 'Distribution Tax Federal'],
            [date_end, cash_monthly_state_penalty * months, 'Distribution Tax State'],
            [date_end, cash_monthly_state_penalty * months_penalties, 'Distribution Early Withdrawal Penalty'],
        ]
        df_cashflows = pd.DataFrame(cashflows_IRA_bank_list, columns=['Date', 'Cashflow Portfolio', 'Description'])[['Date', 'Cashflow Portfolio']]
        df_cashflows = pd.concat([self.cashflows_portfolio, df_cashflows], ignore_index=True).copy()
        df_cashflows = df_cashflows.groupby(['Date'], as_index=False).aggregate({'Cashflow Portfolio': 'sum'})
        df_cashflows = df_cashflows.sort_values(by=['Date'])
        df_cashflows = df_cashflows.round(digits)
        df_cashflows = df_cashflows.reset_index(drop=True)
        self.cashflows_IRA_bank = df_cashflows
    
    def values_from_cashflows(self, cashflows, date_final=None, col_date='Date', col_cashflows='Cashflow', col_values='Value'):
        cashflows = cashflows.copy()
        cashflows = cashflows[cashflows[col_cashflows] != 0]
        cashflows = cashflows.groupby([col_date], as_index=False).aggregate({col_cashflows: 'sum'})
    
        col_gains_date = 'Date'
        col_gains = 'Forward Gain Cumulative'

        if date_final is None:
            mask_cashflows = (cashflows['Date'] <= pd.Timestamp.now()) | True
            mask_valuations = (self.valuations['Date'] <= pd.Timestamp.now()) | True
        else:
            mask_cashflows = (cashflows['Date'] <= date_final)
            mask_valuations = (self.valuations['Date'] <= date_final)

        cashflows = cashflows[mask_cashflows]

        values = self.valuations[[col_gains_date, col_gains, col_values]].copy()
        values = values[mask_valuations]
        
        values[col_cashflows] = 0.
        values[col_values] = 0.
        values = values[[col_gains_date, col_cashflows, col_gains, col_values]].fillna(1.)
        
        for index, row in cashflows.iterrows():
            date = cashflows.loc[index, col_date]
            cashflow = cashflows.loc[index, col_cashflows]
            date_gains = values[values[col_gains_date] >= date][col_gains_date].min()
            mask = values[col_gains_date] == date_gains
            values.loc[mask, col_cashflows] = cashflow
            gain_present = values.loc[mask, col_gains].values[0]
        
            mask = values[col_gains_date] >= date_gains
            cashflows_future = cashflow * values[mask][col_gains]
            values.loc[mask, col_values] += cashflows_future / gain_present
        
        return values
            
    def calculate_values_from_cashflows(self, cashflows=None, date_final=None, debug=0):
        if cashflows is None:
            cashflows = self.cashflows_portfolio

        values = self.values_from_cashflows(cashflows, date_final=date_final, col_date='Date', col_cashflows='Cashflow Portfolio', col_values='Value Portfolio')
        
        return values

    def display_values_from_cashflows(self, cashflows=None, date_final=None, display_max_rows=None, print_results=True, debug=0):
        if date_final is None:
            mask_valuations = (self.valuations['Date'] <= pd.Timestamp.now()) | True
        else:
            mask_valuations = (self.valuations['Date'] <= date_final)

        start = np.datetime64('now')

        values = self.calculate_values_from_cashflows(cashflows=cashflows, date_final=date_final, debug=0)
        
        end = np.datetime64('now')
        elapsed = end - start

        if print_results:
            value_self_final_actual = self.valuations[mask_valuations].iloc[-1]['Value Portfolio']
            value_self_final_projected = values.iloc[-1]['Value Portfolio']
            
            value_delta = (value_self_final_actual - value_self_final_projected).round(2)

            print(f"Final portfolio value determined correctly from cashflows and gains?  {(value_self_final_actual - value_self_final_projected).round(2) == 0}")

        return values

    def display_lost_portfolio_value(self, tax_rate=0.2, date_final=None, display_dataframes=False):
        if date_final is None:
            date_final = self.valuations.iloc[-1]['Date']

        df_cashflows_no_bank = self.calculate_values_from_cashflows(cashflows=self.cashflows_IRA_no_bank_distributions, date_final=self.cashflows_IRA_no_bank_distributions.iloc[-1]['Date'])
        date_beginning = self.cashflows_IRA_no_bank_distributions.iloc[0]['Date'].date()
        date_ending = self.cashflows_IRA_no_bank_distributions.iloc[-1]['Date'].date()

        cashflows_inheritance_distribution = self.cashflows_IRA_no_bank_distributions[self.cashflows_IRA_no_bank_distributions['Description'] == 'Distribution Cash Flow']['Cashflow Portfolio'].sum()
        cashflows_inheritance_federal = self.cashflows_IRA_no_bank_distributions[self.cashflows_IRA_no_bank_distributions['Description'] == 'Distribution Tax Federal']['Cashflow Portfolio'].sum()
        cashflows_inheritance_state = self.cashflows_IRA_no_bank_distributions[self.cashflows_IRA_no_bank_distributions['Description'] == 'Distribution Tax State']['Cashflow Portfolio'].sum()
        cashflows_inheritance_penalty = self.cashflows_IRA_no_bank_distributions[self.cashflows_IRA_no_bank_distributions['Description'] == 'Distribution Early Withdrawal Penalty']['Cashflow Portfolio'].sum()
        cashflows_inheritance_total = self.cashflows_IRA_no_bank_distributions['Cashflow Portfolio'].sum()
        value_inheritance_total = df_cashflows_no_bank.iloc[-1]['Value Portfolio']
        gain_cashflows = value_inheritance_total / cashflows_inheritance_total
        gain_initial = float(df_cashflows_no_bank.tail(1)['Forward Gain Cumulative'].values[0])
        gain_final = float(self.valuations[self.valuations['Date'] <= date_final].tail(1)['Forward Gain Cumulative'].values[0])
        gain_interval = gain_final / gain_initial
        gain_total = gain_cashflows * gain_interval
        value_inheritance_final = value_inheritance_total * gain_interval-cashflows_inheritance_distribution
        gain_distribution = value_inheritance_total / cashflows_inheritance_distribution
        gain_net = value_inheritance_final / cashflows_inheritance_distribution
        value_portfolio = self.valuations[self.valuations['Date'] == date_final].iloc[-1]['Value Portfolio']
        value_portfolio_projected = value_portfolio + value_inheritance_final

        str = f"Projected total cashflows from IRA between {date_beginning:%-d %b %Y} and {date_ending:%-d %b %Y}"
        print(f"{str:<80} = {cashflows_inheritance_total:>13,.2f}")

        str = f"    Distributions to bank accounts"
        print(f"{str:<63} = {cashflows_inheritance_distribution:>13,.2f}")

        str = f"    Federal tax distributions"
        print(f"{str:<63} = {cashflows_inheritance_federal:>13,.2f}")

        str = f"    State tax distributions"
        print(f"{str:<63} = {cashflows_inheritance_state:>13,.2f}")
        print(f"    Early withdrawal penalties from IRA               = {cashflows_inheritance_penalty:>13,.2f}")

        str = f"Projected value of cashflows in IRA on {date_ending:%-d %b %Y}"
        print(f"{str:<63} = {value_inheritance_total:>13,.2f}")

        print()

        str = f"Projected distribution gain between {date_beginning:%-d %b %Y} and {date_ending:%-d %b %Y}"
        print(f"{str:<80} = {gain_distribution:>14.3f}")

        str = f"Investment gain from {date_ending:%-d %b %Y} to {date_final:%-d %b %Y}"
        print(f"{str:<80} = {gain_interval:>14.3f}")

        str = f"Projected distribution gain from {date_beginning:%-d %b %Y} to {date_final:%-d %b %Y}"
        print(f"{str:<80} = {gain_distribution * gain_interval:>14.3f}")

        print()

        str = f"Projected value of cashflows from IRA, invested from {date_ending:%-d %b %Y} to {date_final:%-d %b %Y}"
        print(f"{str:<80} = {value_inheritance_total * gain_interval:>13,.2f}")

        str = f"    Distributions to bank accounts"
        print(f"{str:<63} = {gain_total * cashflows_inheritance_distribution:>13,.2f}")

        str = f"    Federal tax in IRA"
        print(f"{str:<63} = {gain_total * cashflows_inheritance_federal:>13,.2f}")

        str = f"    State tax in IRA"
        print(f"{str:<63} = {gain_total * cashflows_inheritance_state:>13,.2f}")

        str = f"    Early withdrawal penalties from IRA"
        print(f"{str:<63} = {gain_total * cashflows_inheritance_penalty:>13,.2f}")

        print()

        str = f"Projected balance of bank accounts from {date_ending:%-d %b %Y} to {date_final:%-d %b %Y}"
        print(f"{str:<80} = {-cashflows_inheritance_distribution:>13,.2f}")

        str = f"Projected change in net worth from {date_ending:%-d %b %Y} to {date_final:%-d %b %Y}"
        print(f"{str:<80} = {value_inheritance_final:>13,.2f}")
    
        str = f"Projected net gain from {date_beginning:%-d %b %Y} to {date_final:%-d %b %Y}"
        print(f"{str:<80} = {gain_net:>14.3f}")

        str = f"Actual IRA value on {date_final:%-d %b %Y}"
        print(f"{str:<80} = {value_portfolio:>13,.2f}")

        str = f"Projected value of IRA and bank accounts on {date_final:%-d %b %Y}"
        print(f"{str:<80} = {value_portfolio_projected:>13,.2f}")

    def plot_cumulative_gain(self, date_final=None):
        if date_final is None:
            valuations = self.valuations.copy()
        else:
            mask_portfolio = (self.valuations['Date'] <= date_final)
            valuations = self.valuations[mask_portfolio].copy()

        df = valuations[['Date', 'Forward Gain Cumulative']].copy()

        gain_min = df['Forward Gain Cumulative'].min()
        gain_max = df['Forward Gain Cumulative'].max()

        ax = df.plot(x='Date', y='Forward Gain Cumulative', logy=True, grid=True)
        
        gain_max_mantissa = self.fman(gain_max)
        gain_max_exponent = self.fexp(gain_max)
        axis_max_mantissa = int(math.ceil(gain_max_mantissa))
        axis_max = np.maximum(float(axis_max_mantissa * 10**gain_max_exponent), 20)

        match axis_max:
            case 20:
                ax.set_ylim(0.8, 20)
            case 30:
                ax.set_ylim(0.8, 30)
            case 402:
                ax.set_ylim(0.8, 40)
            case 50:
                ax.set_ylim(0.8, 50)
        
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.myLogFormat))
        plt.ylabel ('Forward Gain Cumulative')
    
        ax.grid('on', which='major', axis='both', linewidth=1)
        ax.grid('on', which='minor', axis='both', linewidth=0.3)
        ax.get_legend().remove()
        
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        
        plt.show()

    def plot_portfolio_net(self, date_final=None):
        if date_final is None:
            valuations = self.valuations.copy()
        else:
            mask_portfolio = (self.valuations['Date'] <= date_final)
            valuations = self.valuations[mask_portfolio].copy()

        df = valuations[['Date', 'Net Portfolio']].copy()

        net_min = df['Net Portfolio'].min()
        net_max = df['Net Portfolio'].max()

        ax = df.plot(x='Date', y='Net Portfolio', logy=True, grid=True)
        
        net_max_mantissa = self.fman(net_max)
        net_max_exponent = self.fexp(net_max)
        axis_max_mantissa = int(math.ceil(net_max_mantissa))
        axis_max = np.maximum(float(axis_max_mantissa * 10**net_max_exponent), 20)

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.myLogFormat))
        plt.ylabel ('Net Portfolio')
    
        ax.grid('on', which='major', axis='both', linewidth=1)
        ax.grid('on', which='minor', axis='both', linewidth=0.3)
        ax.get_legend().remove()
        
        ax.xaxis.set_minor_locator(mdates.YearLocator())

        plt.show()

    def myLogFormat(self, y, pos):
        # Find the number of decimal places required
        decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
        # Insert that number into a format string
        formatstring = '{{:,.{:1d}f}}'.format(decimalplaces)
        # Return the formatted tick label
        return formatstring.format(y)
    
    def fexp(self, number):
        (sign, digits, exponent) = Decimal(number).as_tuple()
        return len(digits) + exponent - 1

    def fman(self, number):
        return Decimal(number).scaleb(-self.fexp(number)).normalize()
    
    def sort_by_abs_value(self, date_final=None):
        if date_final is None:
            valuations_local = self.valuations.copy()
        else:
            mask_portfolio = (self.valuations['Date'] <= date_final)
            valuations_local = self.valuations[mask_portfolio].copy()

        columns_dividends = [c for c in valuations_local.columns[2:] if ('Net Dividend' in c)]
        columns_asset = [c for c in valuations_local.columns[2:] if ('Net Fidelity' in c) | ('Net Commission' in c)]
        valuations_total = valuations_local.iloc[-1][columns_asset]
        valuations = pd.DataFrame(valuations_local.iloc[-1][columns_asset])
        valuations.columns = ['Net Final Asset', *valuations.columns[1:]]
        valuations['Net Final Dividends'] = 0.
    
        for idx in valuations_local.iloc[-1][columns_asset].index:
            idx_asset = idx[4:]
            idx_dividend = "Net Dividend:" + idx_asset
            if idx_dividend in columns_dividends:
                valuations.loc[idx, 'Net Final Dividends'] = valuations_local.iloc[-1][idx_dividend]
    
        valuations['Net Final Total'] = valuations['Net Final Dividends'] + valuations['Net Final Asset']
        valuations = valuations.astype(float).round(2)
    
        index_sorted = valuations['Net Final Total'].abs().sort_values(ascending=False).index
        valuations = valuations.reindex(index_sorted)
        valuations['Net Final Total, cumulative'] = valuations['Net Final Total'].cumsum()
        return valuations
        
    def display_sorted_assets(self):
        with pd.option_context('display.max_columns', None, 'display.float_format', '{:,.2f}'.format):
            display(self.sort_by_abs_value().style.format(precision=2, thousands=",", decimal="."))
    
    def display_winners_others(self, date_final=None):
        valuations_sorted = self.sort_by_abs_value(date_final=date_final)
        net_total_cum = valuations_sorted.iloc[-1]['Net Final Total, cumulative']
        index_threshold = valuations_sorted[valuations_sorted['Net Final Total, cumulative'] > net_total_cum].index[0]
        valuations_net = valuations_sorted.loc[:index_threshold]
        valuations_null = valuations_sorted.loc[index_threshold:].iloc[1:]
        valuations_null.loc[:,'Net Final Total, cumulative'] = valuations_null.loc[:,'Net Final Total, cumulative'] - valuations_net.iloc[-1]['Net Final Total, cumulative']
        with pd.option_context('display.max_columns', None, 'display.float_format', '{:,.2f}'.format):
            display(valuations_net)
            display(valuations_null)

    def print_outflows_total(self):
        outflows = -self.valuations[(self.valuations['Cashflow Portfolio'] < 0)]['Cashflow Portfolio'].sum()
        print(f"Total distributions from portfolio = ${outflows:,.2f}")
    
from ipywidgets import interact, IntSlider, Layout
from IPython.display import display

def freeze_header(df, num_rows=30, num_columns=10, step_rows=1,
                  step_columns=1, style_format=None):
    """
    Freeze the headers (column and index names) of a Pandas DataFrame. A widget
    enables to slide through the rows and columns.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame to display
    num_rows : int, optional
        Number of rows to display
    num_columns : int, optional
        Number of columns to display
    step_rows : int, optional
        Step in the rows
    step_columns : int, optional
        Step in the columns

    Returns
    -------
    Displays the DataFrame with the widget
    """
    @interact(last_row=IntSlider(min=min(num_rows, df.shape[0]),
                                 max=df.shape[0],
                                 step=step_rows,
                                 description='rows',
                                 readout=False,
                                 disabled=False,
                                 continuous_update=True,
                                 orientation='horizontal',
                                 slider_color='purple',
                                 layout=Layout(width='100%')),
              last_column=IntSlider(min=min(num_columns, df.shape[1]),
                                    max=df.shape[1],
                                    step=step_columns,
                                    description='columns',
                                    readout=False,
                                    disabled=False,
                                    continuous_update=True,
                                    orientation='horizontal',
                                    slider_color='purple',
                                 layout=Layout(width='100%')))
    def _freeze_header(last_row, last_column):
        if style_format is not None:
            display(df.iloc[max(0, last_row-num_rows):last_row,
                            max(0, last_column-num_columns):last_column].style.format(style_format))
        else:
            display(df.iloc[max(0, last_row-num_rows):last_row,
                            max(0, last_column-num_columns):last_column])

style_format =  { \
                    'Date': '{:%Y-%m-%d}', \
                    'Cashflow Bank': '{:,.2f}', \
                    'Balance Bank': '{:,.2f}', \
                    'Share Jason': '{:.1%}', \
                    'Cashflow Jason': '{:,.2f}', \
                    'Cumulative Cashflow Jason': '{:,.2f}', \
                    'Exclusive Jason': '{:,.2f}', \
                    'Cumulative Exclusive Jason': '{:,.2f}', \
                    'Share Elisa': '{:.1%}', \
                    'Cashflow Elisa': '{:,.2f}', \
                    'Cumulative Cashflow Elisa': '{:,.2f}', \
                    'Exclusive Elisa': '{:,.2f}', \
                    'Cumulative Exclusive Elisa': '{:,.2f}', \
                    'Delta': '{:,.2f}', \
                    'Cumulative Delta': '{:,.2f}', \
                    'Cumulative Inheritance Jason in IRA Jason': '{:,.2f}', \
                    'Cumulative Inheritance Jason in IRA Elisa': '{:,.2f}', \
                    'Cumulative Inheritance Elisa in IRA Jason': '{:,.2f}', \
                    'Cumulative Inheritance Elisa in IRA Elisa': '{:,.2f}', \
                    'Cumulative Cashflow Jason excess': '{:,.2f}', \
                    'Cashflow Portfolio Jason': '{:,.2f}', \
                    'Cumulative Cashflow Portfolio Jason': '{:,.2f}', \
                    'Value Jason': '{:,.2f}', \
                    'Value Elisa': '{:,.2f}', \
                }
