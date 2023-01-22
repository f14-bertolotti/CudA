
.PHONY: game_of_life lenia larger_than_life primordia clean

clean:
	make -C game_of_life clean
	make -C lenia clean
	make -C primordia clean
	make -C larger_than_life clean

game_of_life:
	make -C game_of_life run

primordia:
	make -C primordia run

lenia:
	make -C lenia run

larger_than_life:
	make -C larger_than_life run

