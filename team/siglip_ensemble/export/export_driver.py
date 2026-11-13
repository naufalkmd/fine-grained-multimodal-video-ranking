"""Export, verify in a separate offline process, then package. No training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from export_submission import export_submission, package_submission


def main():
    p=argparse.ArgumentParser()
    for name in ('run-dir','contracts','train-zip','work-dir','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--episodes',type=int,default=834)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if sys.version_info[:2] != (3,12):
        raise RuntimeError('Use the Python 3.12 environment created by the training notebook')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    work=Path(args.work_dir);work.mkdir(parents=True,exist_ok=False)
    artifact=export_submission(args.run_dir,work/'artifact')
    report=output/'offline_verification.json'
    with tempfile.TemporaryDirectory(prefix='empty-hf-',dir=work) as empty:
        env=dict(os.environ,HF_HOME=empty,HF_HUB_CACHE=str(Path(empty)/'hub'),
                 TRANSFORMERS_CACHE=str(Path(empty)/'transformers'),HF_HUB_OFFLINE='1',
                 TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
        # A separate CWD and process make the artifact's local-only dependencies testable.
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('verify_submission.py').resolve()),
             '--artifact-dir',str(artifact),'--run-dir',str(Path(args.run_dir).resolve()),
             '--contracts',str(Path(args.contracts).resolve()),'--train-zip',str(Path(args.train_zip).resolve()),
             '--report',str(report.resolve()),'--episodes',str(args.episodes),'--device',args.device]
        subprocess.run(cmd,cwd=empty,env=env,check=True)
    result=json.loads(report.read_text(encoding='utf-8'))
    if not result['t4_verified']:
        raise RuntimeError('CPU parity report saved, but a free T4 timing check is required before creating the submission ZIP')
    destination=output/'siglip_minimal_3seed_submission.zip'
    package_submission(artifact,destination,report)
    print('\n최종 제출 ZIP:',destination,flush=True)
    print('이 ZIP만 채점 사이트에 직접 업로드하세요. 자동 제출은 하지 않았습니다.',flush=True)
    print(f"검증: {result['episodes']}개 영상, 로딩+추론 {result['total_seconds']:.1f}초. 숨겨진 전체 Test 시간 보장은 아닙니다.",flush=True)


if __name__=='__main__':
    main()
