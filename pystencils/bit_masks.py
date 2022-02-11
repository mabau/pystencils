import sympy as sp
# from pystencils.typing import get_type_of_expression


# noinspection PyPep8Naming
class flag_cond(sp.Function):
    """Evaluates a flag condition on a bit mask, and returns the value of one of two expressions,
    depending on whether the flag is set. 

    Three argument version:
    ```
        flag_cond(flag_bit, mask, expr) = expr if (flag_bit is set in mask) else 0
    ```

    Four argument version:
    ```
        flag_cond(flag_bit, mask, expr_then, expr_else) = expr_then if (flag_bit is set in mask) else expr_else
    ```
    """

    nargs = (3, 4)

    def __new__(cls, flag_bit, mask_expression, *expressions):

        # TODO Jan reintroduce checking
        # flag_dtype = get_type_of_expression(flag_bit)
        # if not flag_dtype.is_int():
        #     raise ValueError('Argument flag_bit must be of integer type.')
        #
        # mask_dtype = get_type_of_expression(mask_expression)
        # if not mask_dtype.is_int():
        #     raise ValueError('Argument mask_expression must be of integer type.')

        return super().__new__(cls, flag_bit, mask_expression, *expressions)

    def to_c(self, print_func):
        flag_bit = self.args[0]
        mask = self.args[1]

        then_expression = self.args[2]

        flag_bit_code = print_func(flag_bit)
        mask_code = print_func(mask)
        then_code = print_func(then_expression)

        code = f"(({mask_code}) >> ({flag_bit_code}) & 1) * ({then_code})"

        if len(self.args) > 3:
            else_expression = self.args[3]
            else_code = print_func(else_expression)
            code += f" + (({mask_code}) >> ({flag_bit_code}) ^ 1) * ({else_code})"

        return code
