import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

//Solution to the coin change problem from : Bytedance
//Z���Ļ���ϵͳ������ֵ1Ԫ��4Ԫ��16Ԫ��64Ԫ����4��Ӳ�ң��Լ���ֵ1024Ԫ��ֽ�ҡ�����СYʹ��1024Ԫ��ֽ�ҹ�����һ����ֵΪ����Ʒ���������������յ�����Ӳ�ң�
public class CoinChange {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		@SuppressWarnings("resource")
		Scanner sc = new Scanner(System.in);
		int price = 1024 - sc.nextInt();
		int res = 0;
		List<Integer> coins = new LinkedList<Integer>();
		coins.add(64);
		coins.add(16);
		coins.add(4);
		coins.add(1);
		Iterator<Integer> iter = coins.iterator();
		while(price != 0){
			int temp = iter.next();
			res += price/temp;
			price = price%temp;
		}
		System.out.println(res);
	}

}
