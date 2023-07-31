from utils.utils import remove_uncomplete_runs, remove_test_dir
try:
    remove_test_dir()
except:
    print("No test dir to remove")
try:
    remove_uncomplete_runs()
except:
    print("Removal failed")