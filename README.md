AUTHOR

name : Sangguk Lee (Luan)
e-mail : sangguk258@gmail.com

INSTRUCTION

0. elasticsearch를 실행합니다.

새 윈도우를 열어서

1. source [VENV]/cv3/bin/activate 로 virtual environment를 활성화합니다.


2.0. 다음 경로에서 모델을 다운로드받아 /model 에 압축을 풉니다.
	http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
	이미 받아놨으므로, 안해도 됩니다.

2. 다음 커맨드로 인덱싱할 수 있습니다.
	python indexer.py [relative path of directory] (-p)

3. 다음 명령어로 서버를 켭니다.
	python findServer.py

외부 컴퓨터 아무거나에서 다음 단계를 해도 됩니다.

4. webFinder 프로젝트를 다운받아서, 압축을 풀고, 경로에 들어가서
	npm install 	로 모듈들을 설치한 뒤
	npm run start로 웹서버를 켭니다.

5. 웹서버의 8000 포트로 접속합니다.