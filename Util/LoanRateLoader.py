import os
import pandas as pd
import baostock as bs
from datetime import datetime, timedelta

class LoanRateLoader:
    def __init__(self, data_dir='./data/macro_data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.bs_logged_in = False
        self.cache_file_loan_rate = os.path.join(self.data_dir, 'loan_rates.csv')

    def login_bs(self):
        if not self.bs_logged_in:
            lg = bs.login()
            if lg.error_code == '0':
                self.bs_logged_in = True
                print("Baostock login successful for LoanRateLoader.")
            else:
                print(f"Baostock login failed for LoanRateLoader: {lg.error_msg}")
                self.bs_logged_in = False
        return self.bs_logged_in

    def logout_bs(self):
        if self.bs_logged_in:
            bs.logout()
            self.bs_logged_in = False
            print("Baostock logout successful for LoanRateLoader.")

    def _fetch_loan_rate_data_from_bs(self, start_date_str, end_date_str):
        """
        Fetches loan rate data from Baostock for a given date range.
        start_date_str, end_date_str: "YYYY-MM-DD"
        """
        if not self.login_bs():
            return None

        print(f"Fetching loan rate data from Baostock: {start_date_str} to {end_date_str}...")
        rs = bs.query_loan_rate_data(start_date=start_date_str, end_date=end_date_str)
        
        if rs.error_code != '0':
            print(f"Baostock query_loan_rate_data failed: {rs.error_msg}")
            return None

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            print(f"No loan rate data returned from Baostock for {start_date_str} to {end_date_str}.")
            return pd.DataFrame()

        result_df = pd.DataFrame(data_list, columns=rs.fields)
        return result_df

    def _process_loan_rate_df(self, df):
        """
        Processes the raw DataFrame from Baostock.
        - Converts 'pubDate' to datetime.
        - Converts rate columns to numeric.
        """
        if df.empty:
            return df
            
        df_copy = df.copy()
        
        df_copy['pubDate'] = pd.to_datetime(df_copy['pubDate'])
        
        rate_columns = [col for col in df_copy.columns if col != 'pubDate']
        
        for col in rate_columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        df_copy.sort_values(by='pubDate', inplace=True)
        df_copy.reset_index(drop=True, inplace=True)
        return df_copy

    def load_loan_rate_data(self, start_date="2005-01-01", end_date=None, use_cache=True):
        """
        Loads loan rate data.
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
        if use_cache and os.path.exists(self.cache_file_loan_rate):
            try:
                cached_df = pd.read_csv(self.cache_file_loan_rate)
                if 'pubDate' in cached_df.columns:
                    cached_df['pubDate'] = pd.to_datetime(cached_df['pubDate'])
                else:
                    print("Cache file for loan rates is missing 'pubDate'. Re-fetching all.")
                    cached_df = pd.DataFrame()

                if cached_df.empty:
                    print(f"Cache file {self.cache_file_loan_rate} is empty.")
                else:
                    print(f"Loaded {len(cached_df)} records from loan rate cache: {self.cache_file_loan_rate}")
            except Exception as e:
                print(f"Error reading loan rate cache file {self.cache_file_loan_rate}: {e}. Will fetch all data.")
                cached_df = pd.DataFrame()
        
        fetch_start_dt = target_start_dt
        
        if not cached_df.empty:
            cached_df.sort_values(by='pubDate', inplace=True)
            last_cached_date = cached_df['pubDate'].max()
            
            next_day_to_fetch = last_cached_date + timedelta(days=1)
            
            if next_day_to_fetch > target_end_dt:
                print("Loan rate cache is up to date. No new data to fetch.")
                final_df = cached_df[(cached_df['pubDate'] >= target_start_dt) & 
                                     (cached_df['pubDate'] <= target_end_dt)].copy()
                final_df.reset_index(drop=True, inplace=True)
                return final_df
            
            fetch_start_dt = next_day_to_fetch
            print(f"Loan rate cache up to {last_cached_date.strftime('%Y-%m-%d')}. Fetching new data from {fetch_start_dt.strftime('%Y-%m-%d')} to {target_end_dt.strftime('%Y-%m-%d')}.")
        else:
            print(f"No loan rate cache found or cache is empty. Fetching data from {fetch_start_dt.strftime('%Y-%m-%d')} to {target_end_dt.strftime('%Y-%m-%d')}.")

        if fetch_start_dt > target_end_dt:
            print(f"Calculated fetch start date {fetch_start_dt.strftime('%Y-%m-%d')} is after target end date {target_end_dt.strftime('%Y-%m-%d')}. No data to fetch.")
            new_data_df_processed = pd.DataFrame()
        else:
            new_data_df_raw = self._fetch_loan_rate_data_from_bs(fetch_start_dt.strftime("%Y-%m-%d"), target_end_dt.strftime("%Y-%m-%d"))
            if new_data_df_raw is not None and not new_data_df_raw.empty:
                new_data_df_processed = self._process_loan_rate_df(new_data_df_raw)
            else:
                new_data_df_processed = pd.DataFrame()

        if not new_data_df_processed.empty:
            combined_df = pd.concat([cached_df, new_data_df_processed], ignore_index=True)
            combined_df.drop_duplicates(subset=['pubDate'], keep='last', inplace=True)
            combined_df.sort_values(by='pubDate', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
        else:
            combined_df = cached_df.copy()

        if not combined_df.empty:
            try:
                combined_df.to_csv(self.cache_file_loan_rate, index=False, encoding='utf-8')
                print(f"Updated loan rate cache file: {self.cache_file_loan_rate} with {len(combined_df)} records.")
            except Exception as e:
                print(f"Error writing to loan rate cache file {self.cache_file_loan_rate}: {e}")
        
        final_df = combined_df[(combined_df['pubDate'] >= target_start_dt) & 
                               (combined_df['pubDate'] <= target_end_dt)].copy()
        final_df.reset_index(drop=True, inplace=True)
        
        self.logout_bs()
        return final_df

if __name__ == '__main__':
    loader = LoanRateLoader()
    
    loan_rates_df = loader.load_loan_rate_data(start_date="2010-01-01")
    if loan_rates_df is not None and not loan_rates_df.empty:
        print("\nLoaded Loan Rate Data:")
        print(loan_rates_df.head())
        print("...")
        print(loan_rates_df.tail())
        print(f"Total records: {len(loan_rates_df)}")
        print(f"Date range: {loan_rates_df['pubDate'].min().strftime('%Y-%m-%d')} to {loan_rates_df['pubDate'].max().strftime('%Y-%m-%d')}")
    else:
        print("No loan rate data loaded.")