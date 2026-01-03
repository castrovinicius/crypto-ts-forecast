"""
This module contains example tests for a Kedro project.
Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py.
"""
import pytest
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality

class TestKedroRun:
    def test_kedro_run_success(self, mocker):
        """Test that the pipeline runs successfully."""
        
        # Mock response with data provided by user (12 days)
        mock_data = [
            [1704326400000,"42845.23000000","44729.58000000","42613.77000000","44151.10000000","48038.06334000",1704412799999,"2095095359.49363480",1819944,"23605.90059000","1030075267.83901300","0"],
            [1704412800000,"44151.10000000","44357.46000000","42450.00000000","44145.11000000","48075.25327000",1704499199999,"2100953924.36871550",2064845,"24015.06426000","1049655019.14000940","0"],
            [1704499200000,"44145.12000000","44214.42000000","43397.05000000","43968.32000000","17835.06144000",1704585599999,"781212883.49160610",956642,"9048.57073000","396341787.36387760","0"],
            [1704585600000,"43968.32000000","44480.59000000","43572.09000000","43929.02000000","23023.85080000",1704671999999,"1014635096.12414820",1109259,"11613.52347000","511871007.52582630","0"],
            [1704672000000,"43929.01000000","47248.99000000","43175.00000000","46951.04000000","72814.57589000",1704758399999,"3298772150.68150740",2364464,"38728.80530000","1755312207.66818340","0"],
            [1704758400000,"46951.04000000","47972.00000000","44748.67000000","46110.00000000","69927.66617000",1704844799999,"3251442072.80839630",2637745,"34825.45429000","1620151226.33626950","0"],
            [1704844800000,"46110.00000000","47695.93000000","44300.36000000","46653.99000000","89911.41203000",1704931199999,"4123809481.71276720",3133588,"46910.79439000","2153126701.35216420","0"],
            [1704931200000,"46654.00000000","48969.48000000","45606.06000000","46339.16000000","87470.32960000",1705017599999,"4105139605.02830200",2998451,"44076.16830000","2070467939.65999250","0"],
            [1705017600000,"46339.16000000","46515.53000000","41500.00000000","42782.73000000","86327.93707000",1705103999999,"3827483735.57605690",2809192,"41707.58066000","1850840878.97782980","0"],
            [1705104000000,"42782.74000000","43257.00000000","42436.12000000","42847.99000000","36118.47464000",1705190399999,"1547370611.55316620",1434243,"17748.09824000","760432218.24545510","0"],
            [1705190400000,"42847.99000000","43079.00000000","41720.00000000","41732.35000000","28228.40894000",1705276799999,"1202212782.25769300",1235727,"13690.30843000","583261659.73857610","0"],
            [1705276800000,"41732.35000000","43400.43000000","41718.05000000","42511.10000000","40269.89303000",1705363199999,"1718982911.10810270",1657611,"19924.76030000","849997411.08758290","0"]
        ]

        mock_response_data = mocker.Mock()
        mock_response_data.json.return_value = mock_data
        mock_response_data.raise_for_status.return_value = None

        # Mock empty response to terminate the loop
        mock_response_empty = mocker.Mock()
        mock_response_empty.json.return_value = []
        mock_response_empty.raise_for_status.return_value = None

        mocker.patch(
            "crypto_ts_forecast.pipelines.data_ingestion.nodes.requests.get",
            side_effect=[mock_response_data, mock_response_empty]
        )

        bootstrap_project(Path.cwd())
    
        # Override test_size_days to 2 days so that with 12 days of data,
        # we have 10 days for training (Prophet needs >= 2)
        extra_params = {"prophet": {"test_size_days": 2}}
        
        with KedroSession.create(project_path=Path.cwd(), runtime_params=extra_params) as session:
            session.run()
