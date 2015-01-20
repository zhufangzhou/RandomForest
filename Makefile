all:
	@cd src; make -w all
debug:
	@cd src; make -w debug
clean:
	@cd src; make -w clean
init:
	@cd src; make -w create_dir
