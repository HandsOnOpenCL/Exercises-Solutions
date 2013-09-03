
#ifdef __APPLE__
    #include <OpenGL/gl.h>
    #include <GLUT/glut.h>
#else
    #include <GL/gl.h>
    #include <GL/glut.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#define MULT 1

GLuint texture;

float tex_points[] = {0.0, 1.0,
                      0.0, 0.0,
                      1.0, 0.0,
                      1.0, 1.0};

float quad_points[] = {-1.0, -1.0,
                       -1.0,  1.0,
                        1.0,  1.0,
                        1.0, -1.0};

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
        glTexCoord2f(tex_points[0],tex_points[1]); glVertex2f(quad_points[0],quad_points[1]);
        glTexCoord2f(tex_points[2],tex_points[3]); glVertex2f(quad_points[2],quad_points[3]);
        glTexCoord2f(tex_points[4],tex_points[5]); glVertex2f(quad_points[4],quad_points[5]);
        glTexCoord2f(tex_points[6],tex_points[7]); glVertex2f(quad_points[6],quad_points[7]);
    glEnd();
    // Display!
    glFlush();
    glDisable(GL_TEXTURE_2D);
}

void press(unsigned char key, int x, int y)
{
#define PAN 0.01
#define ZOOM 2.0
    switch (key)
    {
        case 'w':
            tex_points[1] += PAN;
            tex_points[3] += PAN;
            tex_points[5] += PAN;
            tex_points[7] += PAN;
            break;
        case 'a':
            tex_points[0] += PAN;
            tex_points[2] += PAN;
            tex_points[4] += PAN;
            tex_points[6] += PAN;
            break;
        case 's':
            tex_points[1] -= PAN;
            tex_points[3] -= PAN;
            tex_points[5] -= PAN;
            tex_points[7] -= PAN;
            break;
        case 'd':
            tex_points[0] -= PAN;
            tex_points[2] -= PAN;
            tex_points[4] -= PAN;
            tex_points[6] -= PAN;
            break;
        case 'i':
            quad_points[0] *= ZOOM;
            quad_points[1] *= ZOOM;
            quad_points[2] *= ZOOM;
            quad_points[3] *= ZOOM;
            quad_points[4] *= ZOOM;
            quad_points[5] *= ZOOM;
            quad_points[6] *= ZOOM;
            quad_points[7] *= ZOOM;
            break;
        case 'o':
            quad_points[0] /= ZOOM;
            quad_points[1] /= ZOOM;
            quad_points[2] /= ZOOM;
            quad_points[3] /= ZOOM;
            quad_points[4] /= ZOOM;
            quad_points[5] /= ZOOM;
            quad_points[6] /= ZOOM;
            quad_points[7] /= ZOOM;
            break;
        case 'r':
            tex_points[0] = 0.0;
            tex_points[1] = 1.0;
            tex_points[2] = 0.0;
            tex_points[3] = 0.0;
            tex_points[4] = 1.0;
            tex_points[5] = 0.0;
            tex_points[6] = 1.0;
            tex_points[7] = 1.0;
            quad_points[0] = -1.0;
            quad_points[1] = -1.0;
            quad_points[2] = -1.0;
            quad_points[3] = 1.0;
            quad_points[4] = 1.0;
            quad_points[5] = 1.0;
            quad_points[6] = 1.0;
            quad_points[7] = -1.0;
            break;
    }
    glutPostRedisplay();
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Create the texture itself
    glTexImage2D(GL_TEXTURE_2D, 0, 4, nx, ny, 0, GL_RED, GL_UNSIGNED_BYTE, &board[0]);

    // Set display call back
    glutDisplayFunc(display);
    // Set keyboard call back
    glutKeyboardFunc(press);

    // Show the screen
    glutMainLoop();

    return EXIT_SUCCESS;
}

