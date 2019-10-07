#include "vlsift.h"

void fun_git(int argc, char *argv) {
	if (argc >= 2) {
		printf("From vlsift\n\tNow let show the arguments:\n");

		printf("\tlength of string is %d\n", sizeof(argv));
		if (strlen(argv) >= 5) {
			printf("\t%s\n", argv);
		}
	} else {
		printf("From vlsift\n\tNothing can be shown! Please check your input!!\n");
	}
}