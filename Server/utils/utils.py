import platform

print("OS : "+platform.platform())

def check_os_type():

    if platform.system() == "Darwin":
        return "mac"

    elif platform.system() == "Linux":
        return "linux"


def check_toto_banner(url):
    print(url)






