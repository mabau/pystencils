#include "iacaMarks.h"

int main(int argc, char * argv[]){
	int a = 0;
	for(int i = 0; i < argc+100000; i++){
		IACA_START
		a += i;
	}
	IACA_END
	return a;
}
