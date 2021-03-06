# flake8: noqa: F401
"""
Commands module.
Contains all start-commands, subcommands and CLI Interface creation.

Note: Be careful with file-scoped imports in these subfiles.
    as they are parsed on startup, nothing containing optional modules should be loaded.
"""
from freqtrade.commands.arguments import Arguments
from freqtrade.commands.data_commands import start_download_data
from freqtrade.commands.deploy_commands import (start_create_userdir,
                                                start_new_hyperopt,
                                                start_new_strategy)
from freqtrade.commands.hyperopt_commands import (start_hyperopt_list,
                                                  start_hyperopt_show)
from freqtrade.commands.list_commands import (start_list_exchanges,
                                              start_list_markets,
                                              start_list_strategies,
                                              start_list_timeframes)
from freqtrade.commands.optimize_commands import (start_backtesting,
                                                  start_edge, start_hyperopt)
from freqtrade.commands.pairlist_commands import start_test_pairlist
from freqtrade.commands.plot_commands import (start_plot_dataframe,
                                              start_plot_profit)
from freqtrade.commands.trade_commands import start_trading
