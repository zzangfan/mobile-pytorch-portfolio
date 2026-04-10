#  Cat and Dog Classification AI

## 환경 세팅 (Docker)
이 프로젝트는 GPU 가속과 무제한 공유 메모리를 지원하는 도커 환경에서 구동됩니다.
만약 멀티프로세싱기능을 사용할때 error가 발생하면 아래처럼 docker를 새로 만들어주세요
아래 명령어를 통해 컨테이너를 생성하세요:

```bash
docker run -it --gpus all --ipc=host --name <CONTAINER_NAME> -v "<YOUR_LOCAL_DIR>:/workspace" <IMAGE_NAME> /bin/bash