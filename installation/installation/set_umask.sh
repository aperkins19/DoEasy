#!/bin/sh

# this is run on image build entrypoint and sets the file permissions to read, write for everyone.
umask 777 /bin/bash