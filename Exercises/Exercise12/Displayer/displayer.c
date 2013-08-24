
#include <GL/glut.h>
#include <GL/gl.h>

#include <stdio.h>
#include <stdlib.h>

#define MULT 1

GLuint texture;

void display(void)
{
    // Clear the colour buffer for the new display
    glClear(GL_COLOR_BUFFER_BIT);
    // Use a 2D texture
    glEnable(GL_TEXTURE_2D);
    // The shape we will define should take on the colors of the texture
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
    // Use this specific texture
    glBindTexture(GL_TEXTURE_2D, texture);
    // Draw a rectangle the size of the display
    glBegin(GL_QUADS);
        glTexCoord2f(0.0,1.0); glVertex2f(-1.0,-1.0);
        glTexCoord2f(0.0,0.0); glVertex2f(-1.0,1.0);
        glTexCoord2f(1.0,0.0); glVertex2f(1.0,1.0);
        glTexCoord2f(1.0,1.0); glVertex2f(1.0,-1.0);
    glEnd();
    // Display!
    glFlush();
    glDisable(GL_TEXTURE_2D);
}

int main(int argc, char **argv)
{
    // Check for a .dat file to display
    if (argc < 5)
    {
        printf("Usage: display input.dat input.params winX winY\n");
        printf("\tinput.dat\tPattern to draw\n");
        printf("\tinput.params\tParameter file of the board\n");
        printf("\twinX winY\tSize of the window to display the board\n");
        return EXIT_FAILURE;
    }

    // Board size
    unsigned int nx, ny;

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
        board[x + y * nx] = 255;
    }

    // Get window size
    unsigned int winX = atoi(argv[3]);
    unsigned int winY = atoi(argv[4]);

    // Set up GLUT window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(winX, winY);
    glutCreateWindow("Game of Life Viewer");
    glClearColor (0.0, 0.0, 0.0, 0.0);

    // Create the texture from the board
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // Make sure the texture we are about to create lines up
    // in the GPU memory properly
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // Specify what to do when we stretch/shrink the texture to fit on the screen
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // Create the texture itself
    glTexImage2D(GL_TEXTURE_2D, 0, 4, nx, ny, 0, GL_RED, GL_UNSIGNED_BYTE, &board[0]);

    // Set display call back
    glutDisplayFunc(display);

    // Show the screen
    glutMainLoop();

    return EXIT_SUCCESS;
}

