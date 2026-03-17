from __future__ import annotations

from ast import Dict
import pandas as pd
import gc
import os
import time



class user_profile:
    
    def get_user_id(
        train_dataset_path: str,
        train_df: pd.dataframe
    ):

        if os.path.isfile(train_dataset_path) 












def main() -> None:
    t_start = time.time()
    result = user_profile(
        verbose=True,
    )

    if result:
        cli_print_results(result, t_start)




if __name__ == "__main__":
    main()