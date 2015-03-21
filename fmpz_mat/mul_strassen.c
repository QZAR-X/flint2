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

    Copyright (C) 2015 Konstantin Sofiyuk

******************************************************************************/

#include "fmpz_mat.h"

__inline__ static void 
mat_mul_fix_A_ncol_odd(fmpz_mat_t C, const fmpz_mat_t A, const fmpz_mat_t B)
{
    slong n, m, k, i, j;
    n = A->r & ~WORD(1);
    m = B->c & ~WORD(1);
    k = A->c - 1;

    for (i = 0; i < n; i++) 
    {
        for (j = 0; j < m; j++)
        {
            fmpz_addmul(fmpz_mat_entry(C, i, j),
                        fmpz_mat_entry(A, i, k),
                        fmpz_mat_entry(B, k, j));
        }
    }
}

__inline__ static void 
mat_mul_fix_A_nrow_odd(fmpz_mat_t C, const fmpz_mat_t A, const fmpz_mat_t B)
{
    fmpz_mat_t nA, nC;

    fmpz_mat_window_init(nA, A, A->r - 1, 0, A->r, A->c);
    fmpz_mat_window_init(nC, C, A->r - 1, 0, A->r, C->c);

    fmpz_mat_mul_classical_inline(nC, nA, B);

    fmpz_mat_window_clear(nA);
    fmpz_mat_window_clear(nC);
}

__inline__ static void
mat_mul_fix_B_ncol_odd(fmpz_mat_t C, const fmpz_mat_t A, const fmpz_mat_t B)
{
    fmpz_mat_t nA, nB, nC;
    slong ar = A->r & ~WORD(1);

    fmpz_mat_window_init(nA, A, 0, 0, ar, A->c);
    fmpz_mat_window_init(nB, B, 0, B->c - 1, B->r, B->c);
    fmpz_mat_window_init(nC, C, 0, B->c - 1, ar, C->c);

    fmpz_mat_mul_classical_inline(nC, nA, nB);

    fmpz_mat_window_clear(nA);
    fmpz_mat_window_clear(nB);
    fmpz_mat_window_clear(nC);
}

void
fmpz_mat_mul_strassen(fmpz_mat_t C, const fmpz_mat_t A, const fmpz_mat_t B)
{
    fmpz_mat_t A11, A12, A21, A22, B11, B12, B21, B22, C11, C12, C21, C22;
    fmpz_mat_t Wnk, Wkm, Wnk_window;

    slong ar, ac, bc;
    slong ar_half, ac_half, br_half, bc_half, cr_half, cc_half;

    ar = A->r;
    ac = A->c;
    bc = B->c;

    if (ar * bc <= 32 *  32 || ar <= 4 || ac <= 4 || bc <= 4)
    {
        fmpz_mat_mul_classical_inline(C, A, B);
        return;
    } 

    ar_half = ar >> 1;
    ac_half = ac >> 1;
    br_half = ac_half;
    bc_half = bc >> 1;
    cr_half = ar_half;
    cc_half = bc_half;

    fmpz_mat_window_init(A11, A, 0, 0, ar_half, ac_half);
    fmpz_mat_window_init(A12, A, 0, ac_half, ar_half, 2 * ac_half);
    fmpz_mat_window_init(A21, A, ar_half, 0, 2 * ar_half, ac_half);
    fmpz_mat_window_init(A22, A, ar_half, ac_half, 2 * ar_half, 2 * ac_half);

    fmpz_mat_window_init(B11, B, 0, 0, br_half, bc_half);
    fmpz_mat_window_init(B12, B, 0, bc_half, br_half, 2 * bc_half);
    fmpz_mat_window_init(B21, B, br_half, 0, 2 * br_half, bc_half);
    fmpz_mat_window_init(B22, B, br_half, bc_half, 2 * br_half, 2 * bc_half);

    fmpz_mat_window_init(C11, C, 0, 0, cr_half, cc_half);
    fmpz_mat_window_init(C12, C, 0, cc_half, cr_half, 2 * cc_half);
    fmpz_mat_window_init(C21, C, cr_half, 0, 2 * cr_half, cc_half);
    fmpz_mat_window_init(C22, C, cr_half, cc_half, 2 * cr_half, 2 * cc_half);

    fmpz_mat_init(Wnk, ar_half, FLINT_MAX(ac_half, bc_half));
    fmpz_mat_zero(Wnk);
    fmpz_mat_init(Wkm, br_half, bc_half);
    fmpz_mat_window_init(Wnk_window, Wnk, 0, 0, ar_half, ac_half);

    /* 
        For more details see Table 1 from paper:
        Brice Boyer, Jean-Guillaume Dumas, Clément Pernet, Wei Zhou,
        "Memory efficient sheduling of Strassen-Winograd's matrix multipliation algorithm"
        http://arxiv.org/pdf/0707.2347v5.pdf
    */

    fmpz_mat_sub(Wkm, B22, B12);
    fmpz_mat_sub(Wnk_window, A11, A21);
    fmpz_mat_mul_strassen(C21, Wnk_window, Wkm);
    fmpz_mat_add(Wnk_window, A21, A22);
    fmpz_mat_sub(Wkm, B12, B11);
    fmpz_mat_mul_strassen(C22, Wnk_window, Wkm);
    fmpz_mat_sub(Wkm, B22, Wkm);
    fmpz_mat_sub(Wnk_window, Wnk_window, A11);
    fmpz_mat_mul_strassen(C11, Wnk_window, Wkm);
    fmpz_mat_sub(Wnk_window, A12, Wnk_window);
    fmpz_mat_mul_strassen(C12, Wnk_window, B22);
    fmpz_mat_add(C12, C22, C12);
    fmpz_mat_mul_strassen(Wnk_window, A11, B11);
    fmpz_mat_add(C11, C11, Wnk);
    fmpz_mat_add(C12, C11, C12);
    fmpz_mat_add(C11, C11, C21);
    fmpz_mat_sub(Wkm, Wkm, B21);
    fmpz_mat_mul_strassen(C21, A22, Wkm);
    fmpz_mat_sub(C21, C11, C21);
    fmpz_mat_add(C22, C11, C22);
    fmpz_mat_mul_strassen(C11, A12, B21);
    fmpz_mat_add(C11, C11, Wnk);
    

    if (ar & 1) 
    {
        mat_mul_fix_A_nrow_odd(C, A, B);
    }
    if (bc & 1)
    {
        mat_mul_fix_B_ncol_odd(C, A, B);
    }
    if (ac & 1)
    {
        mat_mul_fix_A_ncol_odd(C, A, B);
    }

    fmpz_mat_window_clear(A11);
    fmpz_mat_window_clear(A12);
    fmpz_mat_window_clear(A21);
    fmpz_mat_window_clear(A22);

    fmpz_mat_window_clear(B11);
    fmpz_mat_window_clear(B12);
    fmpz_mat_window_clear(B21);
    fmpz_mat_window_clear(B22);

    fmpz_mat_window_clear(C11);
    fmpz_mat_window_clear(C12);
    fmpz_mat_window_clear(C21);
    fmpz_mat_window_clear(C22);

    fmpz_mat_window_clear(Wnk_window);

    fmpz_mat_clear(Wnk);
    fmpz_mat_clear(Wkm);
}