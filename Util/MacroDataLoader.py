import os
import pandas as pd
import baostock as bs
from datetime import datetime, timedelta

class MacroDataLoader:
    def __init__(self, data_dir='./data/macro_data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.bs_logged_in = False
        self.cache_file_money_supply = os.path.join(self.data_dir, 'money_supply_monthly.csv')

    def login_bs(self):
        if not self.bs_logged_in:
            lg = bs.login()
            if lg.error_code == '0':
                self.bs_logged_in = True
                print("Baostock login successful.")
            else:
                print(f"Baostock login failed: {lg.error_msg}")
                self.bs_logged_in = False
        return self.bs_logged_in

    def logout_bs(self):
        if self.bs_logged_in:
            bs.logout()
            self.bs_logged_in = False
            print("Baostock logout successful.")

    def _fetch_money_supply_data_from_bs(self, start_date_str, end_date_str):
        """
        Fetches money supply data from Baostock for a given month range.
        start_date_str, end_date_str: "YYYY-MM"
        """
        if not self.login_bs():
            return None

        print(f"Fetching money supply data from Baostock: {start_date_str} to {end_date_str}...")
        rs = bs.query_money_supply_data_month(start_date=start_date_str, end_date=end_date_str)
        
        if rs.error_code != '0':
            print(f"Baostock query_money_supply_data_month failed: {rs.error_msg}")
            return None

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            print(f"No money supply data returned from Baostock for {start_date_str} to {end_date_str}.")
            return pd.DataFrame() # Return empty DataFrame if no data

        result_df = pd.DataFrame(data_list, columns=rs.fields)
        return result_df

    def _process_money_supply_df(self, df):
        """
        Processes the raw DataFrame from Baostock.
        - Converts statYear and statMonth to a 'date' column (YYYY-MM-01).
        - Selects relevant columns.
        - Converts columns to numeric types.
        """
        if df.empty:
            return df
            
        df_copy = df.copy()
        # Create 'date' column from statYear and statMonth, set to first day of month
        df_copy['date'] = pd.to_datetime(df_copy['statYear'] + '-' + df_copy['statMonth'] + '-01')
        
        # Define columns to keep and their types
        # We are interested in m0Month, m1Month, m2Month primarily
        columns_to_keep = ['date', 'statYear', 'statMonth', 
                           'm0Month', 'm0YOY', 'm0ChainRelative',
                           'm1Month', 'm1YOY', 'm1ChainRelative',
                           'm2Month', 'm2YOY', 'm2ChainRelative']
        
        # Ensure all desired columns exist, fill with NaN if not (though Baostock should provide them)
        for col in columns_to_keep:
            if col not in df_copy.columns:
                df_copy[col] = pd.NA # Or np.nan

        df_processed = df_copy[columns_to_keep].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Convert relevant columns to numeric, coercing errors
        numeric_cols = ['m0Month', 'm0YOY', 'm0ChainRelative',
                        'm1Month', 'm1YOY', 'm1ChainRelative',
                        'm2Month', 'm2YOY', 'm2ChainRelative']
        for col in numeric_cols:
            if col in df_processed.columns:
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        df_processed.sort_values(by='date', inplace=True)
        df_processed.reset_index(drop=True, inplace=True)
        return df_processed

    def load_money_supply_data(self, start_year=2005, end_year=None, use_cache=True):
        """
        Loads monthly money supply data (M0, M1, M2).
        Fetches data from Baostock if not cached or if cache is outdated.
        start_year: Integer, e.g., 2005
        end_year: Integer, e.g., current year if None.
        """
        if end_year is None:
            end_year = datetime.now().year
        
        # Define the full desired date range for fetching (monthly)
        # Baostock query is by month string "YYYY-MM"
        target_start_month_str = f"{start_year}-01"
        # For end_date, use the last month of the end_year, or current month if end_year is current year
        current_dt = datetime.now()
        if end_year == current_dt.year:
            target_end_month_str = current_dt.strftime("%Y-%m")
        else:
            target_end_month_str = f"{end_year}-12"

        cached_df = pd.DataFrame()
        if use_cache and os.path.exists(self.cache_file_money_supply):
            try:
                cached_df = pd.read_csv(self.cache_file_money_supply)
                if 'date' in cached_df.columns:
                    cached_df['date'] = pd.to_datetime(cached_df['date'])
                else: # If old cache format without 'date', re-fetch all
                    print("Old cache format detected. Re-fetching all data.")
                    cached_df = pd.DataFrame()
                if cached_df.empty:
                     print(f"Cache file {self.cache_file_money_supply} is empty or unreadable.")
                else:
                    print(f"Loaded {len(cached_df)} records from cache: {self.cache_file_money_supply}")
            except Exception as e:
                print(f"Error reading cache file {self.cache_file_money_supply}: {e}. Will fetch all data.")
                cached_df = pd.DataFrame()
        
        # Determine what data needs to be fetched
        fetch_start_date_str = target_start_month_str
        
        if not cached_df.empty:
            cached_df.sort_values(by='date', inplace=True)
            last_cached_date = cached_df['date'].max()
            # Next month to fetch starts after the last cached month
            next_month_to_fetch = last_cached_date + pd.DateOffset(months=1)
            
            # If next_month_to_fetch is already beyond our target_end_month_str, no new data needed
            if next_month_to_fetch > pd.to_datetime(target_end_month_str + "-01"): # Compare with first day of target end month
                print("Cache is up to date. No new data to fetch.")
                # Filter cached_df to the requested start_year and end_year
                final_df = cached_df[(cached_df['date'] >= pd.to_datetime(f"{start_year}-01-01")) & 
                                     (cached_df['date'] <= pd.to_datetime(f"{end_year}-12-31"))].copy()
                final_df.reset_index(drop=True, inplace=True)
                return final_df
            
            fetch_start_date_str = next_month_to_fetch.strftime("%Y-%m")
            print(f"Cache up to {last_cached_date.strftime('%Y-%m')}. Fetching new data from {fetch_start_date_str} to {target_end_month_str}.")

        # Fetch new data (either all or incremental)
        # Ensure fetch_start_date_str is not later than target_end_month_str
        if pd.to_datetime(fetch_start_date_str + "-01") > pd.to_datetime(target_end_month_str + "-01"):
            print(f"Calculated fetch start date {fetch_start_date_str} is after target end date {target_end_month_str}. No data to fetch.")
            new_data_df_processed = pd.DataFrame()
        else:
            new_data_df_raw = self._fetch_money_supply_data_from_bs(fetch_start_date_str, target_end_month_str)
            if new_data_df_raw is not None and not new_data_df_raw.empty:
                new_data_df_processed = self._process_money_supply_df(new_data_df_raw)
            else:
                new_data_df_processed = pd.DataFrame()

        # Combine cached data with new data
        if not new_data_df_processed.empty:
            combined_df = pd.concat([cached_df, new_data_df_processed], ignore_index=True)
            # Remove duplicates, keeping the last (newest) entry if any overlap in dates (e.g. from re-fetching)
            combined_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
            combined_df.sort_values(by='date', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
        else:
            combined_df = cached_df.copy() # Use copy if no new data

        # Save updated combined data to cache
        if not combined_df.empty:
            try:
                combined_df.to_csv(self.cache_file_money_supply, index=False, encoding='utf-8') # Use utf-8
                print(f"Updated cache file: {self.cache_file_money_supply} with {len(combined_df)} records.")
            except Exception as e:
                print(f"Error writing to cache file {self.cache_file_money_supply}: {e}")
        
        # Filter to the originally requested start_year and end_year
        final_df = combined_df[(combined_df['date'] >= pd.to_datetime(f"{start_year}-01-01")) & 
                               (combined_df['date'] <= pd.to_datetime(f"{end_year}-12-31"))].copy()
        final_df.reset_index(drop=True, inplace=True)
        
        self.logout_bs() # Logout after operations
        return final_df

if __name__ == '__main__':
    loader = MacroDataLoader()
    
    # Example: Load data from 2010 to current year
    money_supply_df = loader.load_money_supply_data(start_year=2005)
    if money_supply_df is not None and not money_supply_df.empty:
        print("\nLoaded Money Supply Data (M0, M1, M2 monthly):")
        print(money_supply_df.head())
        print("...")
        print(money_supply_df.tail())
        print(f"Total records: {len(money_supply_df)}")
        print(f"Date range: {money_supply_df['date'].min()} to {money_supply_df['date'].max()}")
    else:
        print("No money supply data loaded.")

    # Example: Load data for a specific range, e.g., 2018 to 2020
    # money_supply_df_specific = loader.load_money_supply_data(start_year=2018, end_year=2020)
    # if money_supply_df_specific is not None and not money_supply_df_specific.empty:
    #     print("\nLoaded Money Supply Data for 2018-2020:")
    #     print(money_supply_df_specific)
    # else:
    #     print("No money supply data loaded for 2018-2020.")

    # Test fetching all data again if cache exists (should use cache and fetch little/nothing new)
    # print("\n--- Second load (testing cache and incremental update) ---")
    # money_supply_df_again = loader.load_money_supply_data(start_year=2005) # Start from earlier to test full range
    # if money_supply_df_again is not None and not money_supply_df_again.empty:
    #     print("\nRe-loaded Money Supply Data:")
    #     print(money_supply_df_again.tail())
    #     print(f"Total records: {len(money_supply_df_again)}")
    # else:
    #     print("No money supply data re-loaded.")