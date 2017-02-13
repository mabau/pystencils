import llvmlite.ir as ir


class Loop(object):
    def __init__(self, builder, start_val, stop_val, step_val=1, loop_name='loop', phi_name="_phi"):
        self.builder = builder
        self.start_val = start_val
        self.stop_val = stop_val
        self.step_val = step_val
        self.loop_name = loop_name
        self.phi_name = phi_name

    def __enter__(self):
        self.loop_end, self.after, phi = self._for_loop(self.start_val, self.stop_val, self.step_val, self.loop_name,
                                                        self.phi_name)
        return phi

    def _for_loop(self, start_val, stop_val, step_val, loop_name, phi_name):
        # TODO size of int??? unisgned???
        integer = ir.IntType(64)

        # Loop block
        pre_loop_bb = self.builder.block
        loop_bb = self.builder.append_basic_block(name='loop_' + loop_name)

        self.builder.branch(loop_bb)

        # Insert an explicit fall through from the current block to loop_bb
        self.builder.position_at_start(loop_bb)

        # Add phi
        phi = self.builder.phi(integer, name=phi_name)
        phi.add_incoming(start_val, pre_loop_bb)

        loop_end_bb = self.builder.append_basic_block(name=loop_name + "_end_bb")
        self.builder.position_at_start(loop_end_bb)

        next_var = self.builder.add(phi, step_val, name=loop_name + '_next_it')
        cond = self.builder.icmp_unsigned('<', next_var, stop_val, name=loop_name + "_cond")

        after_bb = self.builder.append_basic_block(name=loop_name + "_after_bb")

        self.builder.cbranch(cond, loop_bb, after_bb)
        phi.add_incoming(next_var, loop_end_bb)

        self.builder.position_at_end(loop_bb)

        return loop_end_bb, after_bb, phi

    def __exit__(self, exc_type, exc, exc_tb):
        self.builder.branch(self.loop_end)
        self.builder.position_at_end(self.after)
