# TODO: Write this class to compile all of the data!
class DataAggregator:
    def __init__(self, cy, folder_path=None, r_df=None, u_df=None):
        pass  # Init function will just load in the dataframes through whatever means we decide

    def create_all_cadet_info_df(self):
        """
        This method will take the ROTC and USAFA dataframes we've loaded and create the "All Cadet Info" dataframe.
        This will likely call other class methods in a specific order.
        """
        pass

    def create_cadets_fixed_df(self):
        """
        This method will use the "All Cadet Info" dataframe in conjunction with some supporting material and
        create the cadets fixed dataframe used in the instance file.
        """
        pass

    def compile_problem_instance_file(self):
        """
        This is the main method that does everything. Aggregate data, clean it, and then turn it
        into a problem instance file that can be loaded into the main "CadetCareerProblem" Class
        :return:
        """
        pass