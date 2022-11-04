import os, sys
commit_id = 'e1d8efd8e5469cf865a9db60007a70e3f0cb8778'

maestro_dir = "../maestro"
working_path = os.getcwd()

maestro = os.path.join(maestro_dir, "maestro")
maestro =  os.path.abspath(maestro)
if os.path.exists(maestro_dir) is False:
    os.system("git clone git@github.com:maestro-project/maestro.git {}".format(maestro_dir))
    os.chdir(maestro_dir)
    os.system(f"git checkout e1d8efd8e5469cf865a9db60007a70e3f0cb8778")
    try:
        os.system("scons")
    except:
        "Something wring when building maestro, please check maestro repository installation step"
if os.path.exists(maestro) is False:
    os.chdir(maestro_dir)
    try:
        os.system("scons")
    except:
        "Something wring when building maestro, please check maestro repository installation step"
os.chdir(working_path)
