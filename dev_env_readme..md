Docker container is running with volume mount:
- Container: strategyproblem-strategy-env-1 running
- Opencode: Installed at /root/.opencode/bin/opencode
- Volume: ./:/workspace - changes in container are saved to host


To connect to the container:

docker exec -it strategyproblem-strategy-env-1 /bin/bash

Then inside the container, run:

cd /workspace
opencode