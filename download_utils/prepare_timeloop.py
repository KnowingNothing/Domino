import os, sys

timeloop_dir = "../timeloop"
timeloop_src_dir = os.path.join(timeloop_dir, "src")
working_path = os.getcwd()

if not os.path.exists(timeloop_dir):
    os.system(f"git clone git@github.com:NVlabs/timeloop.git {timeloop_dir}")
    os.chdir(timeloop_src_dir)
    os.system("ln -s ../pat-public/src/pat .")
    os.chdir("..")
    os.system("scons --accelergy -j8")

