from afccp import CadetCareerProblem

instance = CadetCareerProblem(N=20, M=5, P=4, printing=True)
instance.fix_generated_data(printing=False)
instance.classic_hr()
instance.export_data()