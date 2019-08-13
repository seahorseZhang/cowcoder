import java.util.ArrayList;
import java.util.Scanner;

//Solution to maximum distance problem from:��ByteDance
//PΪ�����Ķ�άƽ�������㼯������ P ��ĳ��x�����x���� P ������㶼���� x �����Ϸ������ڣ��������궼����x���������Ϊ�����ġ���������С����ġ���ļ��ϡ���
//���е�ĺ�����������궼���ظ�, �����᷶Χ��[0, 1e9) �ڣ�
//����ͼ��ʵ�ĵ�Ϊ���������ĵ�ļ��ϡ���ʵ�ִ����ҵ����� P �е����� ����� ��ļ��ϲ������
public class MaxDistance {
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		ArrayList<Point> points = new ArrayList<>();
		ArrayList<Point> res = new ArrayList<>();
		for(int i = 0; i< n; i++) {
			Point p = new Point(sc.nextInt(), sc.nextInt());
			points.add(p);
		}
		for(Point p: points) {
			boolean temp = true;
			for(Point p1: points)
				if(p1.x > p.x && p1.y > p.y) {
					temp = false;
					break;
				}
			if(temp) res.add(p);
		}
		
		for(Point p: res) {
			System.out.printf("%d%s%d", p.x, ' ', p.y);
			System.out.println();
		}
		
	}

}

class Point {
	int x ;
	int y;
	Point(int x, int y){
		this.x = x;
		this.y = y;
	}
}
