import dvc.api
from pydrive2.auth import GoogleAuth
import subprocess
import typer

app = typer.Typer()



def authenticate():
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile('auth/credentials.json')  # Load the client secrets file
    gauth.CommandLineAuth()  # This will use the --noauth_local_webserver option


def dvc_push():
    print("Hej")
    result = subprocess.run(['dvc', 'push'])
    if result.returncode == 0:
        print("DVC push successful")
    else:
        print("DVC push failed")
        print(result.stderr)

if __name__ == "__main__":
    authenticate()
    dvc_push()

