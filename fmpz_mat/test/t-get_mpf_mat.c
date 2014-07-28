/*=============================================================================

    This file is part of FLINT.

    FLINT is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    FLINT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with FLINT; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA

=============================================================================*/
/******************************************************************************

    Copyright (C) 2010 William Hart
    Copyright (C) 2011 Fredrik Johansson
    Copyright (C) 2014 Abhinav Baid

******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include "flint.h"
#include "fmpz.h"
#include "fmpz_mat.h"
#include "ulong_extras.h"

int
main(void)
{
    int i;
    FLINT_TEST_INIT(state);


    flint_printf("get_mpf_mat....");
    fflush(stdout);

    /* set entries of an fmpz_mat, convert to mpf_mat and then check that
       the entries remain same */
    for (i = 0; i < 1000 * flint_test_multiplier(); i++)
    {
        fmpz_mat_t A;
        mpf_mat_t B;
        slong j, k;
        slong rows = n_randint(state, 10);
        slong cols = n_randint(state, 10);

        fmpz_mat_init(A, rows, cols);
        mpf_mat_init(B, rows, cols, mpf_get_default_prec());

        for (j = 0; j < rows; j++)
        {
            for (k = 0; k < cols; k++)
            {
                fmpz_set_ui(fmpz_mat_entry(A, j, k), 3 * j + 7 * k);
            }
        }

        fmpz_mat_get_mpf_mat(B, A);

        for (j = 0; j < rows; j++)
        {
            for (k = 0; k < cols; k++)
            {
                if (mpf_cmp_ui(mpf_mat_entry(B, j, k), 3 * j + 7 * k) != 0)
                {
                    flint_printf("FAIL: j = %wd, k = %wd\n", j, k);
                    fmpz_mat_print_pretty(A);
                    mpf_mat_print(B);
                    abort();
                }
            }
        }

        fmpz_mat_clear(A);
        mpf_mat_clear(B);
    }



    FLINT_TEST_CLEANUP(state);
    flint_printf("PASS\n");
    return 0;
}
