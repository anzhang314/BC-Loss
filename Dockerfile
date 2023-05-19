FROM nvcr.io/nvidia/tensorflow:19.10-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
	sudo \
      && \
    rm -rf /var/lib/apt/lists/

RUN pip install torch==1.9.0 \
	matplotlib \
	Cython \
	tqdm \
	pandas \
	scikit-learn \
	scipy \
	reckit \
	numpy==1.16.6

# create a non-root user
# please run id -a to see your own uid and gid, and fill them here
# USERNAME could be any name
ARG USER_UID=1002
ARG USERNAME=appuser
ARG USER_GID=$USER_UID

# Create the user
# group and user have the same name
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -m --uid $USER_UID --gid $USER_GID  $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /workspace

COPY . /workspace/BC-Loss
