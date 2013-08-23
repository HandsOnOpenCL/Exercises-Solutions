
#include <GL/glut.h>
#include <GL/gl.h>

#include <stdio.h>
#include <stdlib.h>

#define MULT 5

GLuint texture;
unsigned int nx, ny;

void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBegin(GL_QUADS);
    /*glTexCoord2f(0.0f, 0.0f); glVertex3f((MULT * nx) / -2.0f, (MULT * ny) / -2.0f, 0.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex3f((MULT * nx) / 2.0f, (MULT * ny) / -2.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex3f((MULT * nx) / 2.0f, (MULT * ny) / 2.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex3f((MULT * nx) / -2.0f, (MULT * ny) / 2.0f, 0.0f);
    */
    glTexCoord2f(0.0,1.0); glVertex2f(-1.0,-1.0);
    glTexCoord2f(0.0,0.0); glVertex2f(-1.0,1.0);
    glTexCoord2f(1.0,0.0); glVertex2f(1.0,1.0);
    glTexCoord2f(1.0,1.0); glVertex2f(1.0,-1.0);
    glEnd();
    glFlush();
    glDisable(GL_TEXTURE_2D);
}

int main(int argc, char **argv)
{
    // Check for a .dat file to display
    if (argc < 3)
    {
        printf("Usage: display input.dat input.params\n");
        return EXIT_FAILURE;
    }

    // Load in the params file
    FILE *fp = fopen(argv[2], "r");
    if (!fp)
    {
        printf("Unable to open params file!\n");
        return EXIT_FAILURE;
    }
    int retval;
    retval = fscanf(fp, "%d\n", &nx);
    retval = fscanf(fp, "%d\n", &ny);

    // Recreate the Board from the .dat file
    GLubyte *board = (GLubyte*)calloc(nx * ny, sizeof(GLubyte));
    unsigned int x, y, s;
    fp = fopen(argv[1], "r");

    while ((retval = fscanf(fp, "%d %d %d\n", &x, &y, &s)) != EOF)
    {
        printf("setting %d, %d\n", x, y);
        board[x + y * nx] = 255;
    }

    /*for (int i = 0; i < nx * 15; i++)
    {
            board[i] = 255;
    }*/



    // Set up GLUT window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(MULT * nx, MULT * ny);
    glutCreateWindow("Game of Life Viewer");
    glClearColor (0.0, 0.0, 0.0, 0.0);
    glShadeModel(GL_FLAT);
    glEnable(GL_DEPTH_TEST);

    // Create the texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, 
                   GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 
                   GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, 4, nx, ny, 0, GL_RED, GL_UNSIGNED_BYTE, &board[0]);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutDisplayFunc(display);
    glutMainLoop();

    return EXIT_SUCCESS;
}

