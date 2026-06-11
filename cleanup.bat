@echo off
echo Deleting patch files...
del /F /Q patch.py patch2.py patch3.py patch4.py patch5.py patch_search.py patch_thumbnails.py

echo Deleting scratch files...
del /F /Q scratch_metrics.py scratch_sim.py scratch_test_aggregation.py scratch_test_search.py

echo Deleting debug files...
del /F /Q debug_aggregation.py debug_overlap.py debug_trace.py

echo Deleting log files...
del /F /Q out.txt out50.txt out_cmd.txt pure_log.txt diag_results.txt diagnostic_output.txt

echo Cleanup complete!
pause
