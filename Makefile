# Makefile

EXE=d2q9-bgk

CC=mpiicc
CFLAGS= -std=c99 -Ofast -ipo -Mipa -ffast-math -xbroadwell -march=broadwell -restrict -w -qvec-report0 -qopt-report0 -qopenmp

# CC=mpicc
# CFLAGS= -std=c99 -Wall -O3 -march=broadwell -Wrestrict -fopenmp -fsanitize=address

LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
# REF_FINAL_STATE_FILE=check/128x128.final_state.dat
# REF_AV_VELS_FILE=check/128x128.av_vels.dat
#REF_FINAL_STATE_FILE=check/128x256.final_state.dat
#REF_AV_VELS_FILE=check/128x256.av_vels.dat
# REF_FINAL_STATE_FILE=check/256x256.final_state.dat
# REF_AV_VELS_FILE=check/256x256.av_vels.dat
REF_FINAL_STATE_FILE=check/1024x1024.final_state.dat
REF_AV_VELS_FILE=check/1024x1024.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
