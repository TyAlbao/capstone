"""Run, verify, or acquire data for the configurable Statcast experiment."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','verify','collect','match'],nargs='?',default='run')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('experiment.json'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('results'))
    parser.add_argument('--refit',action='store_true',help='Refit from 2025 in memory; preserve frozen artifacts and the four-file output contract')
    parser.add_argument('--seasons',type=int,nargs='+',default=[2024,2025,2026])
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if args.command=='run':
        from capstone.pipeline import run
        from capstone.verification import protected_state
        before=protected_state()
        run(json.loads(args.config.read_text()),args.output,args.refit)
        if protected_state()!=before:raise RuntimeError('Protected input changed')
        print(f'Complete: {args.output}; protected inputs unchanged',flush=True)
    elif args.command=='verify':
        from capstone.verification import verify
        verify(args.output)
    elif args.command=='collect':
        if not 1<=args.workers<=12:parser.error('workers must be 1-12')
        from capstone.collection import collect
        collect(args.seasons,args.workers)
    else:
        from capstone.matching import build
        build(args.seasons)

if __name__=='__main__':main()
