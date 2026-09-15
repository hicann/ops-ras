# Operator List

> Description:
>
> - **Operator directory**: The directory name is the operator name in lowercase underscore form. Each directory contains all the deliverables for that operator, including code implementation, examples, documentation, and so on. For the directory structure, see [Project Directory](./install/dir_structure.md).
>
> - **Operator execution hardware unit**: Most operators run on AI Core, and a small number of operators run on AI CPU. By default, operators mentioned in the project generally refer to AI Core operators. For detailed information about AI Core and AI CPU, see "Concept Principles and Terminology > Hardware Architecture and Data Processing Principles" in [Ascend C Operator Development](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC).
> - **Operator interface list**: To facilitate operator invocation, CANN provides a set of C APIs for executing operators, generally prefixed with aclnn. For the full list of interfaces, see [aclnn List](op_api_list.md).
> - **V Version Evolution Description**: Some operators have multiple V versions. When using them, select the highest V version (higher-version operators already include all the capabilities of lower-version operators).

All operator categories and operator lists provided by the project are as follows:

<table><thead>
  <tr>
    <th rowspan="2">Operator Category</th>
    <th rowspan="2">Operator Directory</th>
    <th colspan="2">Operator Implementation</th>
    <th>aclnn Invocation</th>
    <th>Graph Mode Invocation</th>
    <th rowspan="2">Operator Execution Hardware Unit</th>
    <th rowspan="2">Description</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
    <th>op_api</th>
    <th>op_graph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Reliability</td>
    <td><a href="../../reliability/matmul_abft_verify">matmul_abft_verify</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>-</td>
    <td>AI Core</td>
    <td>Performs fault detection on a precomputed matrix multiplication result based on V-ABFT.</td>
  </tr>
</tbody>
</table>
