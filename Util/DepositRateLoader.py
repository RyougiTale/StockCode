import os
import pandas as pd
import baostock as bs
from datetime import datetime, timedelta

class DepositRateLoader:
    def __init__(self, data_dir='./data/macro_data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.bs_logged_in = False
        self.cache_file_deposit_rate = os.path.join(self.data_dir, 'deposit_rates.csv')

    def login_bs(self):
        if not self.bs_logged_in:
            lg = bs.login()
            if lg.error_code == '0':
                self.bs_logged_in = True
                print("Baostock login successful for DepositRateLoader.")
            else:
                print(f"Baostock login failed for DepositRateLoader: {lg.error_msg}")
                self.bs_logged_in = False
        return self.bs_logged_in

    def logout_bs(self):
        if self.bs_logged_in:
            bs.logout()
            self.bs_logged_in = False
            print("Baostock logout successful for DepositRateLoader.")

    def _fetch_deposit_rate_data_from_bs(self, start_date_str, end_date_str):
        """
        Fetches deposit rate data from Baostock for a given date range.
        start_date_str, end_date_str: "YYYY-MM-DD"
        """
        if not self.login_bs():
            return None

        print(f"Fetching deposit rate data from Baostock: {start_date_str} to {end_date_str}...")
        rs = bs.query_deposit_rate_data(start_date=start_date_str, end_date=end_date_str)
        
        if rs.error_code != '0':
            print(f"Baostock query_deposit_rate_data failed: {rs.error_msg}")
            return None

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            print(f"No deposit rate data returned from Baostock for {start_date_str} to {end_date_str}.")
            return pd.DataFrame()

        result_df = pd.DataFrame(data_list, columns=rs.fields)
        return result_df

    def _process_deposit_rate_df(self, df):
        """
        Processes the raw DataFrame from Baostock.
        - Converts 'pubDate' to datetime.
        - Converts rate columns to numeric.
        """
        if df.empty:
            return df
            
        df_copy = df.copy()
        
        # Convert 'pubDate' to datetime
        df_copy['pubDate'] = pd.to_datetime(df_copy['pubDate'])
        
        # Identify rate columns (all columns except pubDate)
        rate_columns = [col for col in df_copy.columns if col != 'pubDate']
        
        for col in rate_columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce') # Coerce errors to NaN
        
        df_copy.sort_values(by='pubDate', inplace=True)
        df_copy.reset_index(drop=True, inplace=True)
        return df_copy

    def load_deposit_rate_data(self, start_date="2005-01-01", end_date=None, use_cache=True):
        """
        Loads deposit rate data.
        Fetches data from Baostock if not cached or if cache is outdated.
        start_date: String, "YYYY-MM-DD"
        end_date: String, "YYYY-MM-DD", or current date if None.
        """
        target_start_dt = pd.to_datetime(start_date)
        if end_date is None:
            target_end_dt = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
        else:
            target_end_dt = pd.to_datetime(end_date)

        cached_df = pd.DataFrame()
        if use_cache and os.path.exists(self.cache_file_deposit_rate):
            try:
                cached_df = pd.read_csv(self.cache_file_deposit_rate)
                if 'pubDate' in cached_df.columns:
                    cached_df['pubDate'] = pd.to_datetime(cached_df['pubDate'])
                else: # Should not happen if saved correctly
                    print("Cache file for deposit rates is missing 'pubDate'. Re-fetching all.")
                    cached_df = pd.DataFrame()

                if cached_df.empty:
                    print(f"Cache file {self.cache_file_deposit_rate} is empty.")
                else:
                    print(f"Loaded {len(cached_df)} records from deposit rate cache: {self.cache_file_deposit_rate}")
            except Exception as e:
                print(f"Error reading deposit rate cache file {self.cache_file_deposit_rate}: {e}. Will fetch all data.")
                cached_df = pd.DataFrame()
        
        fetch_start_dt = target_start_dt
        
        if not cached_df.empty:
            cached_df.sort_values(by='pubDate', inplace=True)
            last_cached_date = cached_df['pubDate'].max()
            
            # Next day to fetch starts after the last cached date
            next_day_to_fetch = last_cached_date + timedelta(days=1)
            
            if next_day_to_fetch > target_end_dt:
                print("Deposit rate cache is up to date. No new data to fetch.")
                final_df = cached_df[(cached_df['pubDate'] >= target_start_dt) & 
                                     (cached_df['pubDate'] <= target_end_dt)].copy()
                final_df.reset_index(drop=True, inplace=True)
                return final_df
            
            fetch_start_dt = next_day_to_fetch
            print(f"Deposit rate cache up to {last_cached_date.strftime('%Y-%m-%d')}. Fetching new data from {fetch_start_dt.strftime('%Y-%m-%d')} to {target_end_dt.strftime('%Y-%m-%d')}.")
        else:
             print(f"No deposit rate cache found or cache is empty. Fetching data from {fetch_start_dt.strftime('%Y-%m-%d')} to {target_end_dt.strftime('%Y-%m-%d')}.")


        # Fetch new data (either all or incremental)
        if fetch_start_dt > target_end_dt:
            print(f"Calculated fetch start date {fetch_start_dt.strftime('%Y-%m-%d')} is after target end date {target_end_dt.strftime('%Y-%m-%d')}. No data to fetch.")
            new_data_df_processed = pd.DataFrame()
        else:
            new_data_df_raw = self._fetch_deposit_rate_data_from_bs(fetch_start_dt.strftime("%Y-%m-%d"), target_end_dt.strftime("%Y-%m-%d"))
            if new_data_df_raw is not None and not new_data_df_raw.empty:
                new_data_df_processed = self._process_deposit_rate_df(new_data_df_raw)
            else:
                new_data_df_processed = pd.DataFrame()

        # Combine cached data with new data
        if not new_data_df_processed.empty:
            combined_df = pd.concat([cached_df, new_data_df_processed], ignore_index=True)
            combined_df.drop_duplicates(subset=['pubDate'], keep='last', inplace=True) # pubDate should be unique
            combined_df.sort_values(by='pubDate', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
        else:
            combined_df = cached_df.copy()

        # Save updated combined data to cache
        if not combined_df.empty:
            try:
                combined_df.to_csv(self.cache_file_deposit_rate, index=False, encoding='utf-8')
                print(f"Updated deposit rate cache file: {self.cache_file_deposit_rate} with {len(combined_df)} records.")
            except Exception as e:
                print(f"Error writing to deposit rate cache file {self.cache_file_deposit_rate}: {e}")
        
        # Filter to the originally requested start_date and end_date
        final_df = combined_df[(combined_df['pubDate'] >= target_start_dt) & 
                               (combined_df['pubDate'] <= target_end_dt)].copy()
        final_df.reset_index(drop=True, inplace=True)
        
        self.logout_bs()
        return final_df

if __name__ == '__main__':
    loader = DepositRateLoader()
    
    # Example: Load data from 2010-01-01 to current date
    deposit_rates_df = loader.load_deposit_rate_data(start_date="2010-01-01")
    if deposit_rates_df is not None and not deposit_rates_df.empty:
        print("\nLoaded Deposit Rate Data:")
        print(deposit_rates_df.head())
        print("...")
        print(deposit_rates_df.tail())
        print(f"Total records: {len(deposit_rates_df)}")
        print(f"Date range: {deposit_rates_df['pubDate'].min().strftime('%Y-%m-%d')} to {deposit_rates_df['pubDate'].max().strftime('%Y-%m-%d')}")
    else:
        print("No deposit rate data loaded.")

    # Example: Load data for a specific range, e.g., 2015-01-01 to 2017-12-31
    # deposit_rates_specific = loader.load_deposit_rate_data(start_date="2015-01-01", end_date="2017-12-31")
    # if deposit_rates_specific is not None and not deposit_rates_specific.empty:
    #     print("\nLoaded Deposit Rate Data for 2015-2017:")
    #     print(deposit_rates_specific)
    # else:
    #     print("No deposit rate data loaded for 2015-2017.")