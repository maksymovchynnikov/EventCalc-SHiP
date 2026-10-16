"""Versioned full production data stored on EOS, outside the source checkout."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, tarfile, tempfile

MANIFEST=Path(__file__).with_name('dataset.json')

def dataset():
    return json.loads(MANIFEST.read_text())

def cache_root():
    return Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'eventcalc-ship'/'production'/dataset()['version']

def resolve_full_data(path=None):
    explicit=path or os.environ.get('EVENTCALC_PRODUCTION_DATA')
    root=Path(explicit).expanduser() if explicit else cache_root()
    if not (root/'results/idm_production/sources/provenance.json').is_file():
        raise FileNotFoundError(f'Full production data are not installed at {root}. '
            'Ordinary simulations use the bundled compact tables. For correlated production '
            'or uncertainty scans, run: python -m eventcalc.production.dark_photon.datasets fetch --ssh-user YOUR_CERN_USERNAME; '
            'or set EVENTCALC_PRODUCTION_DATA to an extracted dataset.')
    return root

def verify(root):
    root=Path(root)
    for name,expected in dataset()['files'].items():
        p=root/name
        if not p.is_file():raise FileNotFoundError(p)
        with p.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:raise ValueError(f'Production-data checksum mismatch: {name}')
    return root

def install_archive(archive,destination=None):
    archive=Path(archive); target=Path(destination) if destination else cache_root()
    with archive.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    spec=dataset()
    if actual!=spec['sha256']:raise ValueError('Production archive checksum mismatch')
    if target.exists():return verify(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.eventcalc-',dir=target.parent) as tmp:
        unpack=Path(tmp)/'data';unpack.mkdir()
        with tarfile.open(archive,'r:gz') as stream:
            for member in stream:
                if not member.isfile() or member.name not in spec['files']:
                    raise ValueError(f'Unexpected archive member: {member.name}')
                p=unpack/member.name
                if not p.resolve().is_relative_to(unpack.resolve()):raise ValueError('Unsafe archive path')
                p.parent.mkdir(parents=True,exist_ok=True)
                with stream.extractfile(member) as src,p.open('wb') as out:shutil.copyfileobj(src,out)
        verify(unpack);unpack.rename(target)
    return target

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['fetch','install','verify','path'])
    p.add_argument('--archive',type=Path);p.add_argument('--destination',type=Path)
    p.add_argument('--ssh-user');p.add_argument('--ssh-control-path')
    a=p.parse_args();spec=dataset();destination=a.destination or cache_root()
    if a.action=='path':print(destination);return
    if a.action=='verify':print(verify(destination));return
    if a.action=='install':
        if a.archive is None:p.error('--archive is required for install')
        print(install_archive(a.archive,destination));return
    if destination.exists():print(verify(destination));return
    host=(a.ssh_user+'@' if a.ssh_user else '')+spec['ssh_host']
    with tempfile.TemporaryDirectory(prefix='eventcalc-download-') as temp:
        archive=Path(temp)/spec['archive'];cmd=['scp']
        if a.ssh_control_path:cmd+=['-o','ControlPath='+a.ssh_control_path]
        cmd += [host+':'+spec['eos_path'],str(archive)]
        subprocess.run(cmd,check=True);print(install_archive(archive,destination))
if __name__=='__main__':main()
