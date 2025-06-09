import os
import paramiko
import json
import stat

def basestation_connect(username, glider):
    """ 
    Connects to the seaglider basestation using SSH and SFTP, retrieves the current directory files.

    Uses the set configuration file based on username to authenticate and connect to the basestation.

    Retrieves the entire 'current' data directory files for the specified glider and saves them locally.
    """
    
    # Change cnf file for unique login to basestation
    # Ex: marqjace uses seaglider_pub_marqjace.cnf
    if username == 'marqjace':
        CONFIG_PATH = r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Python\seaglider_plotting\seaglider_pub_marqjace.cnf'
    else:
        raise Exception(
            f"Username {username} is not recognized.\n" 
            "Contact jace.marquardt@oregonstate.edu to set up your login."
            )
   
    GLIDER = glider

    LOCAL_PATH = f'C:/Users/marqjace/seaglider/{GLIDER}/data/current'
    if not os.path.exists(LOCAL_PATH):
        os.makedirs(LOCAL_PATH)

    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)

    transport = paramiko.Transport((CONFIG['bs']['host'], 22))
    if 'password' in CONFIG['bs'] and CONFIG['bs']['password']:

        # Use password authentication
        transport.connect(username=CONFIG['bs']['username'], password=CONFIG['bs']['password'])
        print(f"Connected to {CONFIG['bs']['host']} as {CONFIG['bs']['username']} using password authentication.")

    elif 'privateKeyFile' in CONFIG['bs']:

        # Use private key authentication
        private_key_path = CONFIG['bs']['privateKeyFile']
        private_key = None
        key_password = CONFIG['bs'].get('privateKeyPassword')  # Optional: add this to your config if needed

        # Try different key types
        try:
            private_key = paramiko.RSAKey.from_private_key_file(private_key_path, password=key_password)
        except paramiko.ssh_exception.SSHException:
            try:
                private_key = paramiko.Ed25519Key.from_private_key_file(private_key_path, password=key_password)
            except paramiko.ssh_exception.SSHException:
                try:
                    private_key = paramiko.DSSKey.from_private_key_file(private_key_path, password=key_password)
                except paramiko.ssh_exception.SSHException:
                    raise Exception("Unsupported private key type or incorrect passphrase.")

        transport.connect(username=CONFIG['bs']['username'], pkey=private_key)
        print(f"Connected to {CONFIG['bs']['host']} as {CONFIG['bs']['username']} using key authentication.")
    else:
        raise Exception(
            f"You must specify some way to authenticate on the remote basestation so I\n"
            f"can log in there. Normally you do this in a basestation configuration\n"
            f".cnf file (you're currently using {CONFIG['bs'].get('cnfFile', 'unknown')} ).\n"
            f"In that file, either specify CONFIG.bs.password, or specify CONFIG.bs.privateKeyFile."
        )

    sftp = paramiko.SFTPClient.from_transport(transport)

    # Change to current directory for the specified glider
    sftp.chdir(f'../gliderjail/home/{GLIDER}/current')

    for file in sftp.listdir():
        local_file_path = os.path.join(LOCAL_PATH, file)
        if os.path.exists(local_file_path):
            print(f"Skipping {file} (already exists).")
            continue
        try:
            # Optionally, check if it's a regular file before downloading
            attr = sftp.stat(file)
            if not stat.S_ISREG(attr.st_mode):
                print(f"Skipping {file} (not a regular file).")
                continue
            print(f"Downloading {file}...")
            sftp.get(file, local_file_path)
        except (IOError, PermissionError, OSError) as e:
            print(f"Skipping {file} (no read permission or other error: {e})")

basestation_connect(username='marqjace', glider='sg266')