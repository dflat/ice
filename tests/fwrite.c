#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 

int main(int argc, char* argv[]) {
    FILE *fp = fopen("test.txt", "w");
    fputs("some text...", fp);
    fclose(fp);
    return 1;
}
