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

/* void fmpz_mat_mul_strassen(fmpz_mat_t C, const fmpz_mat_t A, const fmpz_mat_t B); */

#include "fmpz_mat.h"

/* P = (A11 + A22) * (B11 + B22) */
__inline__ static void
mat_mul_strassen_p1(fmpz_mat_t P,
                    const fmpz_mat_t A11, 
                    const fmpz_mat_t A22, 
                    const fmpz_mat_t B11, 
                    const fmpz_mat_t B22)
{
    fmpz_mat_t F, S;
    fmpz_mat_init(F, A11->r, A11->c);
    fmpz_mat_init(S, B11->r, B11->c);

    fmpz_mat_add(F, A11, A22);
    fmpz_mat_add(S, B11, B22);
    fmpz_mat_mul_strassen(P, F, S);

    fmpz_mat_clear(F);
    fmpz_mat_clear(S);
}

/* P = (A1 + A2) * B */
__inline__ static void
mat_mul_strassen_p2_p5(fmpz_mat_t P,
                       const fmpz_mat_t A1,
                       const fmpz_mat_t A2,
                       const fmpz_mat_t B)
{
    fmpz_mat_t F;
    fmpz_mat_init(F, A1->r, A1->c);

    fmpz_mat_add(F, A1, A2);
    fmpz_mat_mul_strassen(P, F, B);

    fmpz_mat_clear(F);
}

/* P = A * (B1 - B2) */
__inline__ static void
mat_mul_strassen_p3_p4(fmpz_mat_t P,
                       const fmpz_mat_t A,
                       const fmpz_mat_t B1,
                       const fmpz_mat_t B2)
{
    fmpz_mat_t S;
    fmpz_mat_init(S, B1->r, B1->c);

    fmpz_mat_sub(S, B1, B2);
    fmpz_mat_mul_strassen(P, A, S);

    fmpz_mat_clear(S);
}

/* P = (A1 - A2) * (B1 + B2) */
__inline__ static void
mat_mul_strassen_p6_p7(fmpz_mat_t P,
                       const fmpz_mat_t A1,
                       const fmpz_mat_t A2,
                       const fmpz_mat_t B1,
                       const fmpz_mat_t B2)
{
    fmpz_mat_t F, S;
    fmpz_mat_init(F, A1->r, A1->c);
    fmpz_mat_init(S, B1->r, B1->c);

    fmpz_mat_sub(F, A1, A2);
    fmpz_mat_add(S, B1, B2);
    fmpz_mat_mul_strassen(P, F, S);

    fmpz_mat_clear(F);
    fmpz_mat_clear(S);
}

/* C = (P1 + P2) + (P3 - P4), where C have shared memory with P1 */
__inline__ static void
mat_mul_strassen_c11_c22(fmpz_mat_t P1,
                         const fmpz_mat_t P2,
                         const fmpz_mat_t P3,
                         const fmpz_mat_t P4)
{
    fmpz_mat_t S;
    fmpz_mat_init(S, P1->r, P1->c);

    fmpz_mat_add(P1, P1, P2);
    fmpz_mat_sub(S, P3, P4);
    fmpz_mat_add(P1, P1, S);

    fmpz_mat_clear(S);
}


__inline__ static void 
mat_mul_fix_A_ncol_odd(fmpz_mat_t C, const fmpz_mat_t A, const fmpz_mat_t B)
{
    slong n, m, k, i, j;
    n = A->r & (~1);
    m = B->c & (~1);
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

    fmpz_mat_mul_classical(nC, nA, B);

    fmpz_mat_window_clear(nA);
    fmpz_mat_window_clear(nC);
}

__inline__ static void
mat_mul_fix_B_ncol_odd(fmpz_mat_t C, const fmpz_mat_t A, const fmpz_mat_t B)
{
    fmpz_mat_t nA, nB, nC;
    slong ar = A->r & (~1);

    fmpz_mat_window_init(nA, A, 0, 0, ar, A->c);
    fmpz_mat_window_init(nB, B, 0, B->c - 1, B->r, B->c);
    fmpz_mat_window_init(nC, C, 0, B->c - 1, ar, C->c);

    fmpz_mat_mul_classical(nC, nA, nB);

    fmpz_mat_window_clear(nA);
    fmpz_mat_window_clear(nB);
    fmpz_mat_window_clear(nC);
}

void
fmpz_mat_mul_strassen(fmpz_mat_t C, const fmpz_mat_t A, const fmpz_mat_t B)
{
    fmpz_mat_t A11, A12, A21, A22, B11, B12, B21, B22, C11, C12, C21, C22;
    fmpz_mat_t P1, P3, P4;

    slong ar, ac, bc;
    slong ar_half, ac_half, br_half, bc_half, cr_half, cc_half;
    slong pr, pc;

    ar = A->r;
    ac = A->c;
    bc = B->c;

    if (ar * bc < 32 * 32)
    {
        fmpz_mat_mul_classical(C, A, B);
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

    pr = ar_half;
    pc = bc_half;

    fmpz_mat_init(P1, pr, pc);
    fmpz_mat_init(P3, pr, pc);
    fmpz_mat_init(P4, pr, pc);
    
    /* P5 = C12; P2 = C21; P6 = C22; P7 = C11; */
    mat_mul_strassen_p1(P1, A11, A22, B11, B22);

    mat_mul_strassen_p2_p5(C21, A21, A22, B11);
    mat_mul_strassen_p2_p5(C12, A11, A12, B22);

    mat_mul_strassen_p3_p4(P3, A11, B12, B22);
    mat_mul_strassen_p3_p4(P4, A22, B21, B11);

    mat_mul_strassen_p6_p7(C22, A21, A11, B11, B12);
    mat_mul_strassen_p6_p7(C11, A12, A22, B21, B22);

    mat_mul_strassen_c11_c22(C11, P1, P4, C12);
    mat_mul_strassen_c11_c22(C22, P1, P3, C21);

    fmpz_mat_add(C12, C12, P3);
    fmpz_mat_add(C21, C21, P4);

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

    fmpz_mat_clear(P1);
    fmpz_mat_clear(P3);
    fmpz_mat_clear(P4);
}