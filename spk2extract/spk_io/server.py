import sys, os, time, glob

try:
    import paramiko
    HAS_PARAMIKO = True
except ModuleNotFoundError as e:
    paramiko = None
    HAS_PARAMIKO = False


def ssh_connect(host, username, password, verbose=True):
    """
    Connect to a remote server via SSH.

    Establishes an SSH connection to a specified host with a given username and password.
    The function will make up to 30 attempts to connect before giving up.

    Parameters
    ----------
    host : str
        The hostname or IP address of the server to connect to.
    username : str
        The username to use for the SSH connection.
    password : str
        The password to use for the SSH connection.
    verbose : bool, optional
        Whether to print debug information during the connection process.
        Default is True.

    Returns
    -------
    paramiko.SSHClient
        An established SSHClient connection if successful.

    Raises
    ------
    paramiko.AuthenticationException
        If authentication fails.
    Exception
        If connection fails due to reasons other than authentication.
    """
    i = 0
    while True:
        if verbose:
            print("Trying to connect to %s (attempt %i/30)" % (host, i + 1))
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host, username=username, password=password)
            if verbose:
                print("Connected to %s" % host)
            break
        except paramiko.AuthenticationException:
            print("Authentication failed when connecting to %s" % host)
            sys.exit(1)
        except Exception as genexcept:
            print("Could not SSH to %s, waiting for it to start" % host)
            print(genexcept)
            i += 1
            time.sleep(2)
        # If we could not connect within time limit
        if i == 30:
            print("Could not connect to %s. Giving up" % host)
            sys.exit(1)
    return ssh
